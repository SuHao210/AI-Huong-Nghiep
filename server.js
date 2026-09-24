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

function isRetryableGeminiError(error) {
    const text = String(error?.message || error || "");
    return /429|rate.?limit|resource.?exhausted|500|502|503|504|service.?unavailable|unavailable|deadline.?exceeded|temporar/i.test(text);
}

async function retryGemini(fn, attempts = 4) {
    let lastError;
    for (let attempt = 0; attempt < attempts; attempt++) {
        try {
            return await fn();
        } catch (error) {
            lastError = error;
            if (!isRetryableGeminiError(error) || attempt === attempts - 1) throw error;
            const delay = Math.min(9000, 700 * (2 ** attempt)) + Math.floor(Math.random() * 350);
            await sleep(delay);
        }
    }
    throw lastError;
}

function createInteractionWithRetry(request, attempts = 4) {
    return retryGemini(() => ai.interactions.create(request), attempts);
}

function createGenerateContentWithRetry(request, attempts = 4) {
    return retryGemini(() => ai.models.generateContent(request), attempts);
}

function createGenerateContentStreamWithRetry(request, attempts = 4) {
    return retryGemini(() => ai.models.generateContentStream(request), attempts);
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
- Khi đủ thông tin, đề xuất 3-5 nghề CỤ THỂ. BẮT BUỘC trình bày ngay dưới tiêu đề "💼 NGHỀ NGHIỆP ĐÁNG THỬ" (hoặc heading tiếng Anh tương ứng) theo đúng cấu trúc: mỗi nghề bắt đầu bằng một dòng riêng dạng `1. **Tên nghề**`, `2. **Tên nghề**`, ...; tuyệt đối không dùng các câu hành động như "chọn ngay", "thử sức", "ghi lại" làm mục nghề.
- Với mỗi nghề: vì sao phù hợp, dữ kiện từ cuộc trò chuyện, điểm cần phát triển và một cách thử thực tế.
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
                recommendedCareers: [],
                interview: null,
                interviewQuestionCache: {},
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
   TRÍCH XUẤT NGHỀ ĐƯỢC GỢI Ý
   ========================================================= */

function extractRecommendationSection(text) {
    const source = String(text || "");
    const start = source.search(/(?:💼\s*)?(?:NGHỀ NGHIỆP ĐÁNG THỬ|CAREERS WORTH TRYING)/i);
    if (start < 0) return "";
    let section = source.slice(start);
    const stop = section.search(/(?:🧭\s*)?BƯỚC TIẾP THEO|(?:🎯.*?(?:quiz|QUIZ|tìm hiểu sâu hơn|learn more))|AI CAREER LAB/i);
    if (stop > 0) section = section.slice(0, stop);
    return section;
}

function cleanCareerTitle(value) {
    return String(value || "")
        .replace(/^\s*[-•*]+\s*/, "")
        .replace(/\*{1,3}/g, "")
        .replace(/\s+$/, "")
        .replace(/^[-–—:]+\s*/, "")
        .trim();
}

function extractRecommendedCareers(text) {
    const section = extractRecommendationSection(text);
    if (!section) return [];

    const careers = [];
    const seen = new Set();
    const lines = section.split(/\r?\n/);

    for (const line of lines) {
        const match = line.match(/^\s*\d+[.)]\s+(?:\*{1,3})?(.+?)(?:\*{1,3})?\s*$/);
        if (!match) continue;

        const title = cleanCareerTitle(match[1]);
        if (!title || title.length > 140) continue;

        // Chỉ nhận dòng tiêu đề nghề: không lấy bước hành động, thời lượng, hoặc câu hướng dẫn.
        if (/^(hãy|chọn|thử|dành|ghi|quay|bắt đầu|ngày|tuần|tự|mang|đọc|làm|xem|ghi chú|dành ra)\b/i.test(title)) continue;
        if (/\b(ngày|buổi|giờ|phút)\b.*\b(thử|làm|ghi|đọc|bắt đầu)\b/i.test(title)) continue;

        const key = title.toLocaleLowerCase('vi-VN');
        if (seen.has(key)) continue;
        seen.add(key);
        careers.push(title);
        if (careers.length >= 5) break;
    }

    return careers;
}


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
- career_directions: 3-5 TÊN NGHỀ CỤ THỂ, ưu tiên lấy từ danh sách nghề nổi bật đã được hệ thống lưu trong phiên hiện tại.
- Không đưa ngành chung chung nếu đã có nghề cụ thể.
- Mỗi nghề phải là một hướng nghề thực sự, không phải hành động, lời nhắc, bước tiếp theo, hoặc tên tính năng.

DANH SÁCH NGHỀ NỔI BẬT ĐƯỢC HỆ THỐNG LƯU:
${(session.recommendedCareers || []).join(" | ") || "Chưa có danh sách nghề nổi bật."}

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
            }, 4);

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

            const manualInterviewMatch = message.match(/(?:phỏng vấn|phong van)\s+(?:mình|tôi|em)?\s*(?:về|ve)\s+(?:nghề\s+)?(.+?)(?:\.|$)/i);
            const wantsSummary = /\b(tổng hợp|tong hop|gợi ý nghề|goi y nghe|hướng nghề|huong nghe|nên theo nghề|nen theo nghe|nghề gì|nghe gi)\b/i.test(message);

            const activeInterview = session.interview && session.interview.active
                ? session.interview
                : null;

            if (manualInterviewMatch && !activeInterview) {
                session.interview = {
                    active: true,
                    career: manualInterviewMatch[1].trim(),
                    askedQuestions: [],
                    turns: 0,
                    startedAt: Date.now()
                };
            }

            const interviewInstruction = (session.interview && session.interview.active)
                ? `\nĐẶC BIỆT — ĐANG PHỎNG VẤN NGHỀ "${session.interview.career}":
- Đây là cuộc phỏng vấn khám phá riêng nghề này.
- Tin nhắn người dùng hiện tại là CÂU TRẢ LỜI cho câu hỏi trước, không phải yêu cầu tạo lại danh sách nghề.
- Không lặp lại danh sách nghề, không tạo Career Lab, không quay lại hồ sơ.
- Chỉ hỏi ĐÚNG 1 câu tiếp theo.
- Câu tiếp theo phải dựa trực tiếp trên câu trả lời gần nhất.
- Luân phiên kiểu câu hỏi để tránh máy móc; không lặp lại câu đã hỏi.
- Chưa kết luận người dùng hợp/không hợp nghề ở giữa cuộc phỏng vấn.
${Array.isArray(session.interview.askedQuestions) && session.interview.askedQuestions.length
    ? `- Các câu đã hỏi: ${session.interview.askedQuestions.slice(-8).join(" | ")}`
    : ""}
\n`
                : (manualInterviewMatch
                    ? `\nĐẶC BIỆT — BẮT ĐẦU PHỎNG VẤN NGHỀ "${manualInterviewMatch[1].trim()}": Không lặp lại danh sách nghề cũ. Không tạo Career Lab. Chỉ hỏi ĐÚNG 1 câu mở đầu liên quan đến nghề này.\n`
                    : "");

            const summaryInstruction = wantsSummary
                ? `\nĐẶC BIỆT — YÊU CẦU TỔNG HỢP: Hãy hoàn thành trọn vẹn câu trả lời. Nếu đưa nghề nghiệp, phải có heading "💼 NGHỀ NGHIỆP ĐÁNG THỬ" và 3-5 nghề theo đúng định dạng số thứ tự. Không cắt ngang danh sách và không chèn lời dẫn lặp lại.\n`
                : "";

            const chatPrompt = `
${SYSTEM_INSTRUCTION}
${interviewInstruction}${summaryInstruction}
LANGUAGE REQUIREMENT:
- Reply entirely in Vietnamese.
- Keep headings in Vietnamese, including "💼 NGHỀ NGHIỆP ĐÁNG THỬ".
- Đây là lượt chat hiện tại. Hãy trả lời tự nhiên và đi thẳng vào ý.

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
                    maxOutputTokens: (session.interview && session.interview.active) || manualInterviewMatch ? 420 : (wantsSummary ? 1400 : 850)
                }
            }, 4);

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
                    const careers = extractRecommendedCareers(fullText);
                    session.lastRecommendation = {
                        interactionId: null,
                        text: fullText,
                        careers,
                        createdAt: Date.now()
                    };
                    session.recommendedCareers = careers;
                }

                if (session.interview && session.interview.active && fullText.trim()) {
                    const nextQuestion = fullText
                        .replace(/^\s*\*+|\*+\s*$/g, "")
                        .trim();
                    if (nextQuestion) {
                        session.interview.askedQuestions ||= [];
                        if (!session.interview.askedQuestions.some(q =>
                            q.toLocaleLowerCase("vi-VN") === nextQuestion.toLocaleLowerCase("vi-VN")
                        )) {
                            session.interview.askedQuestions.push(nextQuestion);
                            session.interview.askedQuestions =
                                session.interview.askedQuestions.slice(-8);
                        }
                        session.interview.turns = Number(session.interview.turns || 0) + 1;
                    }
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
   PHỎNG VẤN NGHỀ — AI TỰ TẠO CÂU HỎI ĐỂ NGƯỜI DÙNG CHỌN
   ========================================================= */

const CAREER_INTERVIEW_SCHEMA = {
    type: "object",
    properties: {
        questions: {
            type: "array",
            minItems: 5,
            maxItems: 6,
            items: {
                type: "object",
                properties: {
                    type: { type: "string" },
                    question: { type: "string" }
                },
                required: ["type", "question"],
                additionalProperties: false
            }
        }
    },
    required: ["questions"],
    additionalProperties: false
};

const interviewQuestionLimiter = rateLimit({
    windowMs: 10 * 60 * 1000,
    limit: 20,
    keyGenerator: sessionOrIpKey,
    standardHeaders: true,
    legacyHeaders: false,
    message: { error: "Bạn vừa tạo khá nhiều bộ câu hỏi. Vui lòng thử lại sau ít phút." }
});

const interviewStartLimiter = rateLimit({
    windowMs: 10 * 60 * 1000,
    limit: 20,
    keyGenerator: sessionOrIpKey,
    standardHeaders: true,
    legacyHeaders: false,
    message: { error: "Bạn vừa bắt đầu quá nhiều cuộc phỏng vấn. Vui lòng thử lại sau ít phút." }
});

app.post(
    "/api/career-interview-start",
    interviewStartLimiter,
    (req, res) => {
        const sessionId = req.headers["x-session-id"];
        const career = typeof req.body?.career === "string" ? req.body.career.trim() : "";
        const question = typeof req.body?.question === "string" ? req.body.question.trim() : "";

        if (typeof sessionId !== "string" || !sessions.has(sessionId)) {
            return res.status(400).json({ error: "Không tìm thấy cuộc trò chuyện hiện tại." });
        }

        if (!career || career.length > 160 || !question || question.length > 600) {
            return res.status(400).json({ error: "Thông tin bắt đầu phỏng vấn không hợp lệ." });
        }

        const session = sessions.get(sessionId);
        session.lastUsedAt = Date.now();

        const careers = Array.isArray(session.recommendedCareers)
            ? session.recommendedCareers
            : [];

        const normalizedCareer = career.toLocaleLowerCase("vi-VN");
        const isRecommendedCareer = careers.some(item =>
            String(item).toLocaleLowerCase("vi-VN") === normalizedCareer
        );

        if (!isRecommendedCareer) {
            return res.status(403).json({
                error: "Chỉ có thể bắt đầu phỏng vấn với nghề vừa được AI đề xuất."
            });
        }

        session.interview = {
            active: true,
            career,
            askedQuestions: [question],
            turns: 1,
            startedAt: Date.now()
        };

        session.messages ||= [];
        session.messages.push({
            role: "assistant",
            text: question,
            at: Date.now()
        });
        if (session.messages.length > 40) {
            session.messages = session.messages.slice(-40);
        }

        return res.json({ ok: true, career, question });
    }
);

app.post(
    "/api/career-interview-questions",
    interviewQuestionLimiter,
    async (req, res) => {
        const sessionId = req.headers["x-session-id"];
        const career = typeof req.body?.career === "string" ? req.body.career.trim() : "";

        if (typeof sessionId !== "string" || !sessions.has(sessionId)) {
            return res.status(400).json({ error: "Không tìm thấy cuộc trò chuyện hiện tại." });
        }
        if (!career || career.length > 160) {
            return res.status(400).json({ error: "Tên nghề không hợp lệ." });
        }

        const session = sessions.get(sessionId);
        session.lastUsedAt = Date.now();
        session.interviewQuestionCache ||= {};

        const cacheKey = career.toLocaleLowerCase("vi-VN");
        if (Array.isArray(session.interviewQuestionCache[cacheKey]) && !req.body?.refresh) {
            return res.json({ ok: true, career, questions: session.interviewQuestionCache[cacheKey], cached: true });
        }

        const transcript = (session.messages || [])
            .slice(-24)
            .map(m => `${m.role === "user" ? "NGƯỜI DÙNG" : "AI"}: ${String(m.text || "").slice(0, 1600)}`)
            .join("\n\n");

        const prompt = `
Hãy tạo 5-6 câu hỏi để PHỎNG VẤN KHÁM PHÁ nghề "${career}" cho đúng người dùng trong cuộc trò chuyện này.

Mục tiêu là để người dùng tự chọn 1 câu hỏi để bắt đầu trò chuyện, không phải bài kiểm tra kiến thức.

BẮT BUỘC:
- Mỗi câu hỏi chỉ kiểm tra một góc nhìn.
- Các câu phải khác kiểu nhau, ưu tiên: sở thích/điều tò mò; tình huống thực tế; cách giải quyết vấn đề; môi trường làm việc; động lực; cách học/thích nghi.
- Mỗi câu phải cụ thể với nghề "${career}", tránh câu hỏi chung chung kiểu "Bạn có thích nghề này không?".
- Không hỏi lại nguyên văn điều người dùng đã nói.
- Không kết luận người dùng hợp hay không hợp nghề.
- Câu hỏi phải ngắn, tự nhiên và dễ trả lời.
- Chỉ trả JSON theo schema, không markdown.

LỊCH SỬ CUỘC TRÒ CHUYỆN:
${transcript}
`;

        try {
            const response = await createGenerateContentWithRetry({
                model: MODEL,
                contents: prompt,
                config: {
                    responseMimeType: "application/json",
                    responseSchema: CAREER_INTERVIEW_SCHEMA,
                    systemInstruction: "Bạn là AI Hướng Nghiệp. Chỉ tạo câu hỏi khám phá, không chẩn đoán nghề nghiệp."
                }
            }, 4);

            const raw = typeof response?.text === "string" ? response.text.trim() : "";
            let parsed;
            try { parsed = JSON.parse(raw); } catch {
                throw new Error("Gemini trả về bộ câu hỏi không hợp lệ.");
            }

            const questions = Array.isArray(parsed?.questions)
                ? parsed.questions
                    .map(q => ({ type: String(q?.type || "Khám phá"), question: String(q?.question || "").trim() }))
                    .filter(q => q.question)
                    .slice(0, 6)
                : [];

            if (questions.length < 5) {
                throw new Error("Gemini chưa tạo đủ câu hỏi phỏng vấn.");
            }

            session.interviewQuestionCache[cacheKey] = questions;
            return res.json({ ok: true, career, questions, cached: false });
        } catch (error) {
            console.error("Career interview questions error:", error);
            return res.status(503).json({ error: getFriendlyError(error) });
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
        const focusedCareers = Array.isArray(session.recommendedCareers) ? session.recommendedCareers : extractRecommendedCareers(recommendationText);

        // Chỉ mở quiz cho các lĩnh vực được suy ra từ DANH SÁCH NGHỀ nổi bật,
        // không quét toàn bộ đoạn văn để tránh bắt nhầm từ ngoài luồng.
        if (!focusedCareers.length) {
            return res.status(403).json({ error: "Chưa có danh sách nghề nổi bật để tạo quiz." });
        }

        const matches = getMatchingIndustries(focusedCareers.join(" | "));
        if (!matches.includes(industry)) {
            return res.status(403).json({ error: "Nghề nổi bật hiện tại chưa có bộ quiz riêng cho lĩnh vực này." });
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


    if (/429|rate.?limit|resource.?exhausted/i.test(message)) {

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


    if (/500|502|503|504|service.?unavailable|unavailable|high demand|temporar|deadline.?exceeded/i.test(message)) {

        return "AI đang xử lý hơi quá tải hoặc kết nối tạm thời không ổn định. Hệ thống đã tự thử lại; bạn chờ vài giây rồi thử lại nhé.";

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
