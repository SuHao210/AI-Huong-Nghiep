import express from "express";
import dotenv from "dotenv";
import helmet from "helmet";
import rateLimit, { ipKeyGenerator } from "express-rate-limit";
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

    return `ip:${ipKeyGenerator(req.ip)}`;
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

async function createGenerateContentWithRetry(request, attempts = 3) {
    let lastError;
    for (let attempt = 0; attempt < attempts; attempt++) {
        try {
            return await ai.models.generateContent(request);
        } catch (error) {
            lastError = error;
            if (!is429(error) || attempt === attempts - 1) throw error;
            await sleep(700 * (attempt + 1));
        }
    }
    throw lastError;
}

async function createGenerateContentStreamWithRetry(request, attempts = 2) {
    let lastError;
    for (let attempt = 0; attempt < attempts; attempt++) {
        try {
            return await ai.models.generateContentStream(request);
        } catch (error) {
            lastError = error;
            if (!is429(error) || attempt === attempts - 1) throw error;
            await sleep(500 * (attempt + 1));
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
Bạn là Chuyên Gia Hướng Nghiệp AI. Trò chuyện tự nhiên bằng tiếng Việt.
Mục tiêu: giúp người dùng khám phá sở thích, điểm mạnh, cách tư duy, động lực và môi trường làm việc phù hợp; không ép chọn một nghề duy nhất.
- Đây là hội thoại, không phải bài trắc nghiệm. Mỗi lượt chỉ hỏi tối đa 1 câu khi còn thiếu thông tin.
- Không hỏi lại điều người dùng đã nói.
- Khi chưa đủ thông tin, hỏi câu tiếp theo dựa trên câu trả lời gần nhất. Không kết luận quá sớm.
- Mỗi câu hỏi phải có lý do từ câu trả lời trước. Tuyệt đối tránh chuỗi câu hỏi máy móc.
- LUÂN PHIÊN KIỂU CÂU HỎI để cuộc trò chuyện đa dạng: (1) câu hỏi mở về sở thích, (2) đào sâu lý do/động lực, (3) tình huống thực tế, (4) chọn giữa hai cách làm, (5) câu hỏi về môi trường làm việc, (6) cách xử lý thất bại, (7) dự án người dùng muốn tự làm, (8) giá trị và điều họ không muốn đánh đổi, (9) cách họ học một kỹ năng mới. Không cần dùng tất cả; chọn kiểu phù hợp nhất ở từng lượt.
- Tránh hỏi liên tiếp cùng một dạng như "Bạn thích gì?", "Bạn có thích... không?". Nếu người dùng đã nói một sở thích, hãy chuyển sang hỏi về lý do, hành vi, trải nghiệm, mức độ kiên trì hoặc cách họ giải quyết vấn đề.
- Có thể dùng câu hỏi giả định, mini-case hoặc lựa chọn A/B khi chúng giúp phân biệt hai hướng nghề nghiệp. Không chấm đúng/sai trừ khi người dùng yêu cầu.
- Nếu người dùng trả lời quá ngắn, hãy hỏi một câu dễ mở rộng bằng ví dụ cụ thể thay vì lặp lại câu hỏi cũ.
- Nếu người dùng đang hào hứng với một lĩnh vực, hãy đào sâu lĩnh vực đó trước khi mở thêm hướng khác. Nếu họ tỏ ra không thích một hướng, ghi nhận và không cố ép họ.
- Khi đủ thông tin, tóm tắt: sở thích nổi bật, điểm mạnh, kiểu tư duy, động lực, môi trường phù hợp và điều nên phát triển.
- Đề xuất khoảng 3 nghề CỤ THỂ. Với mỗi nghề: vì sao phù hợp, dữ kiện từ cuộc trò chuyện, điểm cần phát triển và một cách thử thực tế.
- Cuối cùng nêu 3 việc nhỏ có thể thử trong 7 ngày.
- Không khẳng định nghề nào là định mệnh; chỉ xem là gợi ý để thử nghiệm.
- Câu hỏi đơn giản: trả lời gọn 2-5 câu, đi thẳng vào ý chính, không mở đầu dài và không lặp lại lời người dùng.
- Với câu hỏi khám phá, có thể trả lời ngắn 1-3 câu rồi hỏi tiếp một câu duy nhất. Với tình huống/case, cho đủ bối cảnh nhưng không biến thành bài thi dài.
- Khi đề xuất nghề, ưu tiên nghề cụ thể và có thể kiểm chứng bằng một trải nghiệm nhỏ. Có thể đề xuất các nghề giao thoa giữa nhiều sở thích thay vì chỉ chọn một nhóm ngành truyền thống.
- Thân thiện, dễ hiểu, không phán xét.
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
                messages: [],
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
   HỒ SƠ HƯỚNG NGHIỆP
   ========================================================= */

const PROFILE_SCHEMA = {
    type: "object",
    properties: {
        interests: { type: "array", items: { type: "string" } },
        strengths: { type: "array", items: { type: "string" } },
        thinking_style: { type: "string" },
        motivations: { type: "string" },
        work_environment: { type: "string" },
        growth_areas: { type: "array", items: { type: "string" } },
        career_directions: { type: "array", items: { type: "string" } }
    },
    required: [
        "interests",
        "strengths",
        "thinking_style",
        "motivations",
        "work_environment",
        "growth_areas",
        "career_directions"
    ],
    additionalProperties: false
};

app.post(
    "/api/profile",
    async (req, res) => {

        const sessionId = req.headers["x-session-id"];

        if (
            typeof sessionId !== "string" ||
            !sessions.has(sessionId)
        ) {
            return res.status(400).json({
                error: "Chưa có phiên trò chuyện. Hãy nhắn với AI trước."
            });
        }

        const session = sessions.get(sessionId);
        session.lastUsedAt = Date.now();

        if (!Array.isArray(session.messages) || !session.messages.some(m => m.role === "user")) {
            return res.status(400).json({
                error: "Hãy nhắn với AI ít nhất một lần để hồ sơ có dữ liệu."
            });
        }

        let releaseGeminiSlot;

        try {

            releaseGeminiSlot = await acquireGeminiSlot();

            const prompt = `
Hãy tạo HỒ SƠ HƯỚNG NGHIỆP TẠM THỜI cho chính người dùng trong cuộc trò chuyện hiện tại.

Chỉ dùng thông tin đã xuất hiện trong cuộc trò chuyện. Không bịa dữ kiện.
Nếu dữ liệu chưa đủ, hãy ghi nhận là chưa đủ thay vì suy đoán.
Hồ sơ dùng để khám phá, không phải chẩn đoán hay kết luận nghề nghiệp.

Yêu cầu:
- interests: 2-5 sở thích/tín hiệu đã được người dùng thể hiện.
- strengths: 2-5 điểm mạnh có bằng chứng từ cách người dùng trả lời.
- thinking_style: 1-2 câu ngắn.
- motivations: 1-2 câu ngắn.
- work_environment: 1-2 câu ngắn.
- growth_areas: 2-4 điểm nên phát triển.
- career_directions: 3-5 hướng nghề cụ thể đang đáng khám phá, chỉ khi có tín hiệu trong cuộc trò chuyện.

Trả JSON đúng schema, không thêm markdown.
`;

            const transcript = (session.messages || [])
                .slice(-30)
                .map((m) => `${m.role === "user" ? "NGƯỜI DÙNG" : "AI"}: ${m.text}`)
                .join("\n\n");

            const profileResponse = await createGenerateContentWithRetry({
                model: MODEL,
                contents: `${prompt}

LỊCH SỬ CUỘC TRÒ CHUYỆN:
${transcript}`,
                config: {
                    responseMimeType: "application/json",
                    responseSchema: PROFILE_SCHEMA,
                    systemInstruction:
                        "Bạn là AI phân tích hồ sơ hướng nghiệp. Chỉ sử dụng dữ kiện có trong transcript. Không bịa, không chẩn đoán, không khẳng định nghề nghiệp là định mệnh. Trả JSON hợp lệ theo schema."
                }
            }, 2);

            const raw = typeof profileResponse?.text === "string"
                ? profileResponse.text.trim()
                : "";

            let profile;
            try {
                profile = JSON.parse(raw);
            } catch {
                throw new Error("Gemini trả về hồ sơ không hợp lệ.");
            }

            if (
                !profile ||
                !Array.isArray(profile.interests) ||
                !Array.isArray(profile.strengths) ||
                typeof profile.thinking_style !== "string" ||
                typeof profile.motivations !== "string" ||
                typeof profile.work_environment !== "string" ||
                !Array.isArray(profile.growth_areas) ||
                !Array.isArray(profile.career_directions)
            ) {
                throw new Error("Dữ liệu hồ sơ không đúng định dạng.");
            }

            return res.json({ profile });

        } catch (error) {

            console.error("Profile error:", error);

            return res.status(503).json({
                error: getFriendlyError(error)
            });

        } finally {

            releaseGeminiSlot?.();

        }
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

        session.messages ||= [];
        session.messages.push({
            role: "user",
            text: message,
            at: Date.now()
        });
        if (session.messages.length > 40) session.messages = session.messages.slice(-40);

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
               TẠO CONTEXT CHAT
               ------------------------- */

            const transcript = (session.messages || [])
                .slice(-20)
                .map(m => `${m.role === "user" ? "NGƯỜI DÙNG" : "AI"}: ${String(m.text || "").slice(0, 1400)}`)
                .join("\n\n");

            const chatPrompt = `
${SYSTEM_INSTRUCTION}

LANGUAGE REQUIREMENT:
- Reply entirely in Vietnamese.
- Keep headings in Vietnamese, including "💼 NGHỀ NGHIỆP ĐÁNG THỬ".
- Đây là lượt chat hiện tại. Hãy trả lời tự nhiên, ngắn gọn và đi thẳng vào ý.
- Nếu đang khám phá hướng nghiệp, thường chỉ hỏi một câu tiếp theo.

LỊCH SỬ GẦN ĐÂY:
${transcript}

TIN NHẮN MỚI NHẤT CỦA NGƯỜI DÙNG:
${message}
`;

            /*
             * Chat dùng generateContentStream thay vì interactions streaming.
             * Cách này không phụ thuộc previous_interaction_id nên tránh lỗi
             * 400 khi interaction cũ không còn hợp lệ, đồng thời vẫn stream
             * chữ ra giao diện ngay khi Gemini bắt đầu trả lời.
             */
            const stream = await createGenerateContentStreamWithRetry({
                model: MODEL,
                contents: chatPrompt,
                config: {
                    systemInstruction: "Bạn là AI hướng nghiệp thân thiện, thực tế và luôn trả lời bằng tiếng Việt.",
                    thinkingConfig: { thinkingLevel: "low" },
                    maxOutputTokens: 500
                }
            }, 2);

            let fullText = "";

            for await (const chunk of stream) {
                const chunkText = typeof chunk?.text === "string" ? chunk.text : "";
                if (!chunkText) continue;
                fullText += chunkText;
                sendEvent({ type: "text", text: chunkText });
            }

            if (!fullText.trim()) {
                throw new Error("Gemini không trả về nội dung.");
            }

            /* -------------------------
               LƯU CONVERSATION ID
               ------------------------- */

            if (fullText) {
                session.lastAssistantText = fullText;
                session.messages ||= [];
                session.messages.push({
                    role: "assistant",
                    text: fullText,
                    at: Date.now()
                });
                if (session.messages.length > 40) session.messages = session.messages.slice(-40);
                if (/(NGHỀ NGHIỆP ĐÁNG THỬ|CAREERS WORTH TRYING)/i.test(fullText)) {
                    session.lastRecommendation = {
                        interactionId: null,
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

const NEW_INDUSTRY_RULES = {
    law: {
        title: "Luật & Pháp lý",
        terms: ["law", "legal", "luật", "pháp lý", "luật sư", "lawyer", "legal counsel", "paralegal"]
    },
    education: {
        title: "Giáo dục & Tâm lý",
        terms: ["education", "psychology", "giáo dục", "tâm lý", "teacher", "giáo viên", "psychologist", "counselor"]
    },
    architecture: {
        title: "Kiến trúc & Thiết kế",
        terms: ["architecture", "design", "kiến trúc", "thiết kế", "architect", "kiến trúc sư", "designer", "interior designer"]
    },
    environment: {
        title: "Môi trường & Nông nghiệp",
        terms: ["environment", "agriculture", "môi trường", "nông nghiệp", "environmental analyst", "agronomy", "sustainability", "gis"]
    },
    tourism: {
        title: "Du lịch & Nhà hàng - Khách sạn",
        terms: ["tourism", "hospitality", "du lịch", "nhà hàng", "khách sạn", "hotel", "tour guide", "event", "revenue management"]
    }
};
Object.assign(INDUSTRY_RULES, NEW_INDUSTRY_RULES);

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
        if (!/(NGHỀ NGHIỆP ĐÁNG THỬ|CAREERS WORTH TRYING)/i.test(recommendationText)) {
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
- Ngôn ngữ đầu ra: Tiếng Việt. Cả câu hỏi, 4 lựa chọn và explanation phải dùng tiếng Việt.
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
