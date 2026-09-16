import express from "express";
import dotenv from "dotenv";
import helmet from "helmet";
import rateLimit from "express-rate-limit";
import crypto from "crypto";

import { GoogleGenAI } from "@google/genai";


/* =========================================================
   CẤU HÌNH
   ========================================================= */

dotenv.config();

const app = express();

const PORT =
    Number(process.env.PORT) || 3000;

/*
 * Mặc định dùng Gemini 3.5 Flash-Lite để ưu tiên độ trễ thấp
 * và thông lượng cao cho lớp học đông người.
 *
 * Nếu sau này muốn đổi model, chỉ cần đặt GEMINI_MODEL trên Render.
 */
const MODEL =
    process.env.GEMINI_MODEL ||
    "gemini-3.5-flash-lite";

const API_KEY =
    process.env.GEMINI_API_KEY;


/* =========================================================
   KIỂM TRA API KEY
   ========================================================= */

if (!API_KEY) {

    console.error(
        "❌ Không tìm thấy GEMINI_API_KEY trong file .env"
    );

    process.exit(1);

}


/* =========================================================
   GEMINI CLIENT
   ========================================================= */

const ai =
    new GoogleGenAI({
        apiKey: API_KEY
    });


/* =========================================================
   MIDDLEWARE
   ========================================================= */

app.use(
    helmet({
        contentSecurityPolicy: false
    })
);

app.use(
    express.json({
        limit: "20kb"
    })
);

app.use(
    express.static("public")
);


/* =========================================================
   RATE LIMIT
   =========================================================

   Mỗi IP tối đa 30 request / 10 phút.

   Bạn có thể tăng sau.
   ========================================================= */

/*
 * Rate limit theo SESSION thay vì chỉ theo IP.
 *
 * Điều này rất quan trọng khi 40 học sinh cùng dùng chung
 * Wi-Fi của lớp: tất cả có thể có cùng public IP.
 */
function sessionOrIpKey(req) {
    const sessionId = req.headers["x-session-id"];

    if (
        typeof sessionId === "string" &&
        sessionId.length >= 10 &&
        sessionId.length <= 100
    ) {
        return `session:${sessionId}`;
    }

    return `ip:${req.ip}`;
}

const chatLimiter =
    rateLimit({

        windowMs:
            10 * 60 * 1000,

        /*
         * Mỗi trình duyệt/session được 30 tin nhắn / 10 phút.
         * Không còn giới hạn 30 tin nhắn cho cả lớp dùng chung Wi-Fi.
         */
        limit: 30,

        keyGenerator:
            sessionOrIpKey,

        standardHeaders: true,

        legacyHeaders: false,

        message: {
            error:
                "Bạn gửi khá nhiều tin nhắn. Vui lòng thử lại sau ít phút."
        }

    });


const personalizedQuizLimiter =
    rateLimit({
        windowMs: 60 * 60 * 1000,

        limit: 3,

        keyGenerator:
            sessionOrIpKey,

        standardHeaders: true,

        legacyHeaders: false,

        message: {
            error:
                "Bạn đã tạo quá nhiều quiz riêng trong thời gian ngắn. Vui lòng thử lại sau."
        }
    });

/*
 * Giới hạn số tác vụ Gemini chạy đồng thời trên một instance.
 * Các request còn lại được xếp hàng ngắn; nếu hàng đầy sẽ từ chối
 * thay vì để máy chủ tạo một loạt request cùng lúc tới Gemini.
 */
/*
 * Cho phép nhiều học sinh được xử lý cùng lúc.
 * Gemini vẫn là nơi quyết định hạn mức RPM/TPM thực tế của project.
 */
const MAX_GEMINI_CONCURRENCY = 16;
const MAX_GEMINI_QUEUE = 64;
let activeGeminiRequests = 0;
const geminiQueue = [];

function acquireGeminiSlot() {
    if (activeGeminiRequests < MAX_GEMINI_CONCURRENCY) {
        activeGeminiRequests++;
        return Promise.resolve(() => releaseGeminiSlot());
    }

    if (geminiQueue.length >= MAX_GEMINI_QUEUE) {
        const error = new Error("SERVER_BUSY");
        error.code = "SERVER_BUSY";
        return Promise.reject(error);
    }

    return new Promise((resolve) => {
        geminiQueue.push(resolve);
    });
}

function releaseGeminiSlot() {
    const next = geminiQueue.shift();
    if (next) {
        next(() => releaseGeminiSlot());
        return;
    }
    activeGeminiRequests = Math.max(0, activeGeminiRequests - 1);
}

async function withGeminiSlot(task) {
    const release = await acquireGeminiSlot();
    try {
        return await task();
    } finally {
        release();
    }
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function is429(error) {
    const text = String(error?.message || error || "");
    return /429|rate.?limit|resource.?exhausted/i.test(text);
}

async function createInteractionWithRetry(request, attempts = 2) {
    let lastError;
    for (let attempt = 0; attempt < attempts; attempt++) {
        try {
            return await ai.interactions.create(request);
        } catch (error) {
            lastError = error;
            if (!is429(error) || attempt === attempts - 1) throw error;
            await sleep(600 * (attempt + 1));
        }
    }
    throw lastError;
}


/* =========================================================
   SESSION
   =========================================================

   Bản này dùng RAM để thử nghiệm.

   Mỗi người dùng có một session riêng.

   sessionId
        ↓
   lastInteractionId
        ↓
   Gemini nhớ cuộc hội thoại
   ========================================================= */

const sessions =
    new Map();


/* =========================================================
   SYSTEM INSTRUCTION
   ========================================================= */

const SYSTEM_INSTRUCTION = `

Bạn là "Chuyên Gia Hướng Nghiệp AI".

Bạn đang trò chuyện trực tiếp với một người đang muốn
khám phá sở thích, năng lực và hướng nghề nghiệp phù hợp.

MỤC TIÊU:

Không phải ép người dùng chọn một nghề duy nhất.

Mục tiêu là giúp họ hiểu bản thân hơn và tìm ra
những hướng nghề nghiệp đáng để thử nghiệm.

==================================================
NGUYÊN TẮC HỘI THOẠI
==================================================

Đây là một cuộc trò chuyện tự nhiên.

KHÔNG phải bài trắc nghiệm.

KHÔNG có giới hạn cứng về số câu hỏi.

Không được hỏi liên tục một danh sách câu hỏi.

Mỗi lần người dùng trả lời, hãy đọc kỹ câu trả lời
và quyết định câu hỏi tiếp theo dựa trên thông tin
mới nhất.

Không hỏi lại thông tin người dùng đã nói rõ.

==================================================
KHÁM PHÁ SỞ THÍCH
==================================================

Hãy tìm hiểu:

- Người dùng thích làm gì.
- Họ thường làm gì khi rảnh.
- Việc gì khiến họ mất cảm giác về thời gian.
- Họ thích tạo ra thứ gì.
- Họ thích giải quyết vấn đề gì.
- Điều gì khiến họ tò mò.
- Họ thích làm một mình hay cùng người khác.

Đừng vội biến một sở thích thành một nghề.

Ví dụ:

"Thích game"
không có nghĩa
"phải làm lập trình viên game".

"Thích bóng đá"
không có nghĩa
"phải làm cầu thủ".

Hãy tìm hiểu lý do phía sau sở thích.

==================================================
KHÁM PHÁ NĂNG LỰC
==================================================

Tùy theo cuộc trò chuyện, hãy tìm hiểu một số yếu tố:

- Tư duy logic.
- Khả năng phân tích.
- Sáng tạo.
- Giao tiếp.
- Làm việc nhóm.
- Làm việc độc lập.
- Giải quyết vấn đề.
- Khả năng thích nghi.
- Kiên trì.
- Chủ động.
- Khả năng chịu áp lực.
- Khả năng tổ chức.
- Khả năng lãnh đạo.

Không cần hỏi tất cả.

Chỉ hỏi những yếu tố có liên quan.

==================================================
TÌNH HUỐNG
==================================================

Khi phù hợp, hãy đưa ra các tình huống thực tế.

Ví dụ:

"Nếu bạn đang làm một dự án và kế hoạch ban đầu
không hiệu quả, bạn sẽ làm gì?"

Hoặc:

"Nếu hai thành viên trong nhóm bất đồng ý kiến,
bạn sẽ xử lý thế nào?"

Hoặc:

"Nếu một video bạn đầu tư rất nhiều thời gian
nhưng có rất ít người xem, bạn sẽ làm gì tiếp?"

Những câu hỏi này nhằm hiểu cách người dùng suy nghĩ,
không phải để chấm đúng/sai.

==================================================
KHÔNG KẾT LUẬN QUÁ SỚM
==================================================

Nếu người dùng mới nói một hoặc hai sở thích,
chưa được đưa ra danh sách nghề nghiệp dài.

Hãy tiếp tục khám phá.

Nếu thông tin đã đủ rõ,
hãy chủ động kết luận.

Không cần hỏi đủ một số lượng câu cố định.

==================================================
KHI ĐÃ ĐỦ THÔNG TIN
==================================================

Khi cảm thấy đã có đủ thông tin,
hãy nói rằng bạn đã có đủ cơ sở để phác họa
hướng nghề nghiệp.

Sau đó đưa ra:

🧭 HỒ SƠ HƯỚNG NGHIỆP

- Sở thích nổi bật.
- Điểm mạnh.
- Kiểu tư duy.
- Động lực.
- Môi trường làm việc phù hợp.
- Điều nên phát triển thêm.

Sau đó:

💼 NGHỀ NGHIỆP ĐÁNG THỬ

Đề xuất khoảng 3 nghề cụ thể.

Không nói chung chung như:

"IT"
"kinh doanh"
"truyền thông"

Hãy cụ thể như:

- Data Analyst.
- UX/UI Designer.
- Game Designer.
- Gameplay Programmer.
- Sports Data Analyst.
- Content Strategist.
- Digital Marketing Specialist.
- Product Designer.

Tùy vào thông tin thực tế.

==================================================
MỖI NGHỀ
==================================================

Với mỗi nghề:

1. Vì sao phù hợp.
2. Thông tin nào trong cuộc trò chuyện dẫn tới
   gợi ý này.
3. Điểm nào người dùng cần cải thiện.
4. Mức độ phù hợp:

- Rất phù hợp.
- Khá phù hợp.
- Có tiềm năng.

Không được giả vờ rằng đây là kết quả khoa học
chính xác tuyệt đối.

Không nói:

"Bạn chắc chắn phải làm nghề này."

Hãy nói:

"Nghề này đáng để bạn thử."

==================================================
LỘ TRÌNH
==================================================

Cuối cùng đưa ra:

🚀 3 VIỆC NÊN THỬ NGAY

Mỗi hướng nghề chính nên có:

- Một việc thử trong 7 ngày.
- Một kỹ năng nên học.
- Một dự án nhỏ để kiểm chứng xem người dùng
  có thật sự thích công việc đó hay không.

==================================================
NẾU CHƯA ĐỦ
==================================================

Chỉ hỏi một câu tiếp theo.

Câu hỏi phải tự nhiên.

Câu hỏi phải dựa vào câu trả lời gần nhất.

Không hỏi lại thông tin đã có.

==================================================
PHONG CÁCH
==================================================

Thân thiện.

Tự nhiên.

Không phán xét.

Không làm người dùng cảm thấy đang thi.

Không dùng thuật ngữ quá khó.

Nói tiếng Việt.

Không cần lúc nào cũng dùng emoji.

Nếu người dùng trả lời ngắn,
hãy giúp họ mở rộng câu trả lời bằng câu hỏi dễ.

==================================================
TỐI ƯU TỐC ĐỘ
==================================================

Ưu tiên trả lời gọn, rõ và đi thẳng vào ý chính.

Nếu câu hỏi đơn giản, chỉ cần 2-5 câu hoặc một câu hỏi
tiếp theo phù hợp.

Không viết phần mở đầu dài dòng.

Không lặp lại toàn bộ những gì người dùng vừa nói.

==================================================
QUAN TRỌNG
==================================================

Đừng nói rằng bạn đang "theo dõi số câu hỏi".

Đừng nói rằng bạn phải hỏi đủ 3 câu.

Bạn được phép hỏi nhiều hoặc ít.

Bạn chỉ kết luận khi thông tin đủ.

`;


/* =========================================================
   TẠO SESSION ID
   ========================================================= */

function createSessionId() {

    return crypto.randomUUID();

}


/* =========================================================
   LẤY SESSION
   ========================================================= */

function getSession(req, res) {

    let sessionId =
        req.headers["x-session-id"];


    if (
        typeof sessionId !== "string" ||
        sessionId.length < 10 ||
        sessionId.length > 100
    ) {

        sessionId =
            createSessionId();

    }


    /*
     * Gửi session ID về frontend.
     */

    res.setHeader(
        "X-Session-Id",
        sessionId
    );


    if (!sessions.has(sessionId)) {

        sessions.set(
            sessionId,
            {
                lastInteractionId: null,
                lastAssistantText: "",
                lastRecommendation: null,
                createdAt: Date.now(),
                lastUsedAt: Date.now()
            }
        );

    }


    const session =
        sessions.get(sessionId);


    session.lastUsedAt =
        Date.now();


    return session;

}


/* =========================================================
   DỌN SESSION CŨ
   ========================================================= */

setInterval(
    () => {

        const now =
            Date.now();

        const MAX_AGE =
            2 * 60 * 60 * 1000;


        for (
            const [
                id,
                session
            ] of sessions
        ) {

            if (
                now -
                session.lastUsedAt >
                MAX_AGE
            ) {

                sessions.delete(id);

            }

        }

    },
    15 * 60 * 1000
);


/* =========================================================
   HEALTH CHECK
   ========================================================= */

app.get(
    "/api/health",
    (req, res) => {

        res.json({

            ok: true,

            model: MODEL,

            message:
                "AI Hướng Nghiệp backend đang hoạt động."

        });

    }
);


/* =========================================================
   CHAT
   ========================================================= */

app.post(
    "/api/chat",
    chatLimiter,
    async (req, res) => {

        const message =
            typeof req.body?.message === "string"
                ? req.body.message.trim()
                : "";


        /* -------------------------
           KIỂM TRA INPUT
           ------------------------- */

        if (!message) {

            return res.status(400).json({

                error:
                    "Tin nhắn không được để trống."

            });

        }


        if (message.length > 2000) {

            return res.status(400).json({

                error:
                    "Tin nhắn quá dài. Vui lòng nhập tối đa 2000 ký tự."

            });

        }


        /* -------------------------
           SESSION
           ------------------------- */

        const session =
            getSession(req, res);

        let releaseGeminiSlot;
        try {
            releaseGeminiSlot = await acquireGeminiSlot();
        } catch (error) {
            return res.status(503).json({
                error: getFriendlyError(error)
            });
        }


        /* -------------------------
           SSE HEADERS
           ------------------------- */

        res.status(200);

        res.setHeader(
            "Content-Type",
            "text/event-stream; charset=utf-8"
        );

        res.setHeader(
            "Cache-Control",
            "no-cache, no-transform"
        );

        res.setHeader(
            "Connection",
            "keep-alive"
        );

        res.setHeader(
            "X-Accel-Buffering",
            "no"
        );


        if (
            typeof res.flushHeaders === "function"
        ) {

            res.flushHeaders();

        }


        const sendEvent =
            (payload) => {

                res.write(
                    `data: ${JSON.stringify(payload)}\n\n`
                );

            };


        try {

            /* -------------------------
               TẠO REQUEST
               ------------------------- */

            const request = {

                model:
                    MODEL,

                input:
                    message,

                system_instruction:
                    SYSTEM_INSTRUCTION,

                generation_config: {

                    /*
                     * Low để giảm độ trễ.
                     *
                     * Khi muốn AI suy luận sâu hơn,
                     * có thể đổi thành "medium".
                     */

                    thinking_level:
                        "low"

                },

                stream:
                    true

            };


            /*
             * Nếu đã có hội thoại,
             * Gemini tự giữ lịch sử thông qua
             * previous_interaction_id.
             */

            if (
                session.lastInteractionId
            ) {

                request.previous_interaction_id =
                    session.lastInteractionId;

            }


            /* -------------------------
               GỌI GEMINI STREAM
               ------------------------- */

            const stream =
                await createInteractionWithRetry(request, 2);


            let newInteractionId =
                null;


            let fullText = "";


            /* -------------------------
               ĐỌC STREAM
               ------------------------- */

            for await (
                const event of stream
            ) {

                const eventType =
                    event.type ||
                    event.event_type;


                /* ---------------------
                   INTERACTION CREATED
                   --------------------- */

                if (
                    eventType ===
                    "interaction.created"
                ) {

                    newInteractionId =
                        event
                            ?.interaction
                            ?.id ||
                        null;

                }


                /* ---------------------
                   TEXT DELTA
                   --------------------- */

                if (
                    eventType ===
                    "step.delta"
                ) {

                    const delta =
                        event.delta;


                    if (
                        delta?.type ===
                        "text"
                    ) {

                        const chunk =
                            delta.text ||
                            "";


                        if (chunk) {

                            fullText +=
                                chunk;


                            sendEvent({

                                type:
                                    "text",

                                text:
                                    chunk

                            });

                        }

                    }

                }


                /* ---------------------
                   ERROR
                   --------------------- */

                if (
                    eventType ===
                    "interaction.error"
                ) {

                    const errorMessage =
                        event
                            ?.error
                            ?.message ||
                        "Gemini interaction failed.";

                    sendEvent({
                        type: "error",
                        message: getFriendlyError({ message: errorMessage })
                    });

                    return res.end();
                }

            }


            /* -------------------------
               LƯU CONVERSATION ID
               ------------------------- */

            if (
                newInteractionId &&
                fullText
            ) {

                session.lastInteractionId =
                    newInteractionId;
                session.lastAssistantText =
                    fullText;
                if (/NGHỀ NGHIỆP ĐÁNG THỬ/i.test(fullText)) {
                    session.lastRecommendation = {
                        interactionId: newInteractionId,
                        text: fullText,
                        createdAt: Date.now()
                    };
                }

            }


            /* -------------------------
               HOÀN TẤT
               ------------------------- */

            sendEvent({

                type:
                    "done"

            });


            res.end();

        }


        catch (error) {

            console.error(
                "Gemini error:",
                error
            );


            /*
             * Nếu stream chưa gửi header lỗi
             * thì gửi lỗi dạng SSE.
             */

            sendEvent({

                type:
                    "error",

                message:
                    getFriendlyError(error)

            });


            res.end();

        } finally {
            releaseGeminiSlot?.();
        }

    }
);


/* =========================================================
   QUIZ RIÊNG TỪ CUỘC TRÒ CHUYỆN
   ========================================================= */

const INDUSTRY_RULES = {
    it: {
        title: "Công nghệ thông tin",
        terms: [
            "data analyst", "data scientist", "software engineer",
            "software developer", "lập trình viên", "developer",
            "ux/ui designer", "ui/ux", "game designer",
            "gameplay programmer", "product designer", "devops",
            "cybersecurity", "an ninh mạng", "kỹ sư phần mềm",
            "kiểm thử phần mềm", "qa engineer"
        ]
    },
    business: {
        title: "Kinh doanh & Tài chính",
        terms: [
            "business analyst", "financial analyst", "finance", "tài chính",
            "kinh doanh", "sales", "bán hàng", "business development",
            "chuyên viên kinh doanh", "phân tích kinh doanh", "ngân hàng",
            "accountant", "kế toán", "investment"
        ]
    },
    health: {
        title: "Y tế & Chăm sóc sức khỏe",
        terms: [
            "bác sĩ", "điều dưỡng", "dược sĩ", "pharmacist", "y tế",
            "healthcare", "chăm sóc sức khỏe", "kỹ thuật viên xét nghiệm",
            "vật lý trị liệu", "physiotherapist", "nutritionist", "dinh dưỡng"
        ]
    },
    engineering: {
        title: "Kỹ thuật & Công nghệ",
        terms: [
            "kỹ sư", "engineering", "kỹ thuật", "cơ khí", "điện",
            "điện tử", "tự động hóa", "robotics", "civil engineer",
            "xây dựng", "cơ điện tử", "mechatronics", "mechanical"
        ]
    },
    marketing: {
        title: "Marketing & Truyền thông",
        terms: [
            "marketing", "digital marketing", "content strategist",
            "content creator", "truyền thông", "social media", "copywriter",
            "seo", "branding", "quảng cáo", "pr", "quan hệ công chúng",
            "creator", "biên tập viên"
        ]
    }
};

const PERSONAL_QUIZ_SCHEMA = {
    type: "object",
    properties: {
        questions: {
            type: "array",
            minItems: 20,
            maxItems: 20,
            items: {
                type: "object",
                properties: {
                    question: { type: "string" },
                    options: {
                        type: "array",
                        minItems: 4,
                        maxItems: 4,
                        items: { type: "string" }
                    },
                    answer: { type: "integer", minimum: 0, maximum: 3 },
                    explanation: { type: "string" }
                },
                required: ["question", "options", "answer", "explanation"],
                additionalProperties: false
            }
        }
    },
    required: ["questions"],
    additionalProperties: false
};

function getMatchingIndustries(text) {
    const lower = String(text || "").toLowerCase();
    return Object.entries(INDUSTRY_RULES)
        .filter(([, rule]) => rule.terms.some(term => lower.includes(term)))
        .map(([key]) => key);
}

function getInteractionOutputText(interaction) {
    if (typeof interaction?.output_text === "string") {
        return interaction.output_text;
    }
    const outputs = interaction?.outputs || [];
    return outputs
        .map(output => output?.text || output?.content?.map?.(x => x?.text || "").join("") || "")
        .join("")
        .trim();
}

app.post(
    "/api/personalized-quiz",
    personalizedQuizLimiter,
    async (req, res) => {
        const sessionId = req.headers["x-session-id"];
        const industry = typeof req.body?.industry === "string" ? req.body.industry : "";
        const careerLabel = typeof req.body?.careerLabel === "string" ? req.body.careerLabel : "";

        if (typeof sessionId !== "string" || !sessions.has(sessionId)) {
            return res.status(400).json({ error: "Không tìm thấy cuộc trò chuyện hiện tại." });
        }

        if (!INDUSTRY_RULES[industry]) {
            return res.status(400).json({ error: "Lĩnh vực quiz không hợp lệ." });
        }

        const session = sessions.get(sessionId);
        const recommendation = session.lastRecommendation;
        const recommendationText = recommendation?.text || "";

        // Server-side enforcement: the button can only work when the latest AI
        // response actually reached its career recommendation section and
        // mentioned a profession supported by the selected quiz family.
        if (!/NGHỀ NGHIỆP ĐÁNG THỬ/i.test(recommendationText)) {
            return res.status(403).json({ error: "Quiz riêng chỉ mở sau khi AI đưa ra gợi ý nghề nghiệp." });
        }

        const matches = getMatchingIndustries(recommendationText);
        if (!matches.includes(industry)) {
            return res.status(403).json({ error: "Nghề được gợi ý chưa có bộ quiz riêng cho lĩnh vực này." });
        }

        try {
            const prompt = `
Hãy tạo một bộ QUIZ HƯỚNG NGHIỆP RIÊNG gồm ĐÚNG 20 câu cho người dùng hiện tại.

Lĩnh vực được gợi ý: ${INDUSTRY_RULES[industry].title}
Nghề/lĩnh vực hiển thị trên nút: ${careerLabel || INDUSTRY_RULES[industry].title}

YÊU CẦU QUAN TRỌNG:
- Dựa vào TOÀN BỘ ngữ cảnh cuộc trò chuyện trước đó và đặc điểm người dùng đã chia sẻ.
- Quiz phải cá nhân hóa: tình huống, cách hỏi và trọng tâm phải liên quan tới sở thích, điểm mạnh, cách suy nghĩ, động lực và điều người dùng đã nói.
- Không hỏi lại nguyên văn các câu trong cuộc trò chuyện.
- Không biến quiz thành bài kiểm tra kiến thức chuyên ngành nặng.
- 20 câu, mỗi câu có đúng 4 lựa chọn.
- answer là chỉ số 0,1,2,3 của đáp án đúng.
- explanation ngắn, dễ hiểu, giải thích vì sao đáp án đúng phù hợp với tình huống; không phán rằng người dùng chắc chắn hợp nghề.
- Các đáp án nên có độ phân biệt, không để đáp án đúng luôn ở cùng một vị trí.
- Không thêm markdown, không thêm văn bản ngoài JSON.
- Quiz chỉ mang tính khám phá và tham khảo.
`;

            const interaction = await withGeminiSlot(() =>
                createInteractionWithRetry({
                    model: MODEL,
                    previous_interaction_id: recommendation.interactionId,
                    input: prompt,
                    response_format: {
                        type: "text",
                        mime_type: "application/json",
                        schema: PERSONAL_QUIZ_SCHEMA
                    },
                    generation_config: {
                        thinking_level: "low",
                        max_output_tokens: 6500
                    },
                    // Gemini requires store=true whenever previous_interaction_id is used.
                    // This quiz interaction is intentionally NOT copied into the main chat session
                    // (we do not update session.lastInteractionId below), so the conversation flow
                    // remains separate while still using the previous recommendation as context.
                    store: true
                }, 2)
            );

            const outputText = getInteractionOutputText(interaction);
            let parsed;
            try {
                parsed = JSON.parse(outputText);
            } catch {
                return res.status(502).json({ error: "AI trả về quiz không đúng định dạng. Vui lòng thử lại." });
            }

            if (!Array.isArray(parsed?.questions) || parsed.questions.length !== 20) {
                return res.status(502).json({ error: "AI chưa tạo đủ 20 câu quiz. Vui lòng thử lại." });
            }

            const questions = parsed.questions.map((q, index) => ({
                question: String(q.question || `Câu ${index + 1}`),
                options: Array.isArray(q.options) ? q.options.slice(0, 4).map(String) : [],
                answer: Number(q.answer),
                explanation: String(q.explanation || "")
            }));

            if (questions.some(q => q.options.length !== 4 || !Number.isInteger(q.answer) || q.answer < 0 || q.answer > 3)) {
                return res.status(502).json({ error: "AI trả về một câu hỏi không hợp lệ. Vui lòng thử lại." });
            }

            res.json({
                ok: true,
                title: `Quiz riêng: ${INDUSTRY_RULES[industry].title}`,
                questions
            });
        } catch (error) {
            console.error("Personalized quiz error:", error);
            res.status(503).json({ error: getFriendlyError(error) });
        }
    }
);


/* =========================================================
   RESET CHAT
   ========================================================= */

app.post(
    "/api/reset",
    (req, res) => {

        const sessionId =
            req.headers["x-session-id"];


        if (
            typeof sessionId === "string"
        ) {

            sessions.delete(
                sessionId
            );

        }


        res.json({

            ok:
                true

        });

    }
);


/* =========================================================
   LỖI DỄ HIỂU
   ========================================================= */

function getFriendlyError(error) {

    const message =
        error?.message ||
        "Lỗi không xác định.";

    if (error?.code === "SERVER_BUSY" || message === "SERVER_BUSY") {
        return "Máy chủ đang có nhiều người dùng cùng lúc. Vui lòng thử lại sau ít giây.";
    }

    if (
        message.includes("API key") ||
        message.includes("API_KEY")
    ) {

        return (
            "API Key của server không hợp lệ. " +
            "Hãy kiểm tra GEMINI_API_KEY trong file .env."
        );

    }


    if (
        message.includes("429")
    ) {

        return (
            "Gemini API đang giới hạn lượt sử dụng. " +
            "Vui lòng thử lại sau."
        );

    }


    if (
        message.includes("403")
    ) {

        return (
            "Gemini API từ chối quyền truy cập. " +
            "Hãy kiểm tra API key và project Google."
        );

    }


    if (
        message.includes("404")
    ) {

        return (
            "Gemini tạm thời không xử lý được yêu cầu. " +
            "Vui lòng thử lại sau ít giây."
        );

    }


    return message;

}


/* =========================================================
   START SERVER
   ========================================================= */

app.listen(
    PORT,
    () => {

        console.log("");
        console.log(
            "===================================="
        );

        console.log(
            "🧭 AI HƯỚNG NGHIỆP"
        );

        console.log(
            "===================================="
        );

        console.log(
            `🌐 http://localhost:${PORT}`
        );

        console.log(
            `🤖 Model: ${MODEL}`
        );

        console.log(
            "🔐 API key: server-side"
        );

        console.log(
            "===================================="
        );

        console.log("");

    }
);
