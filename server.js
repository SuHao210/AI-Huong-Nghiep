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

const profileLimiter =
    rateLimit({
        windowMs: 10 * 60 * 1000,
        limit: 12,
        keyGenerator: sessionOrIpKey,
        standardHeaders: true,
        legacyHeaders: false,
        message: {
            error:
                "Bạn cập nhật hồ sơ hơi nhiều trong thời gian ngắn. Vui lòng thử lại sau ít phút."
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

function getErrorText(error) {
    try {
        return String(error?.message || error || "");
    } catch {
        return "";
    }
}

function isRetryableGeminiError(error) {
    const text = getErrorText(error);
    return /(?:429|rate.?limit|resource.?exhausted|500|502|503|504|internal|service.?unavailable|unavailable|temporar|deadline.?exceeded|timeout|econnreset|socket|overloaded|high demand)/i.test(text);
}

function retryDelay(attempt) {
    const base = Math.min(8000, 700 * (2 ** attempt));
    const jitter = Math.floor(Math.random() * 350);
    return base + jitter;
}

async function runWithRetry(task, attempts = 4) {
    let lastError;
    for (let attempt = 0; attempt < attempts; attempt++) {
        try {
            return await task();
        } catch (error) {
            lastError = error;
            if (!isRetryableGeminiError(error) || attempt === attempts - 1) {
                throw error;
            }
            await sleep(retryDelay(attempt));
        }
    }
    throw lastError;
}

async function createInteractionWithRetry(request, attempts = 4) {
    return runWithRetry(() => ai.interactions.create(request), attempts);
}

async function createGenerateContentWithRetry(request, attempts = 4) {
    return runWithRetry(() => ai.models.generateContent(request), attempts);
}

async function createGenerateContentStreamWithRetry(request, attempts = 4) {
    return runWithRetry(() => ai.models.generateContentStream(request), attempts);
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
Bạn là Chuyên Gia Hướng Nghiệp AI. Trò chuyện tự nhiên, rõ ràng và thực tế bằng tiếng Việt.
Mục tiêu: giúp người dùng khám phá sở thích, điểm mạnh, cách tư duy, động lực, môi trường làm việc và các hướng nghề đáng thử; không ép người dùng chọn một nghề duy nhất.

QUY TẮC HỘI THOẠI:
- Đây là hội thoại khám phá, không phải bài trắc nghiệm cứng.
- Khi còn thiếu thông tin, mỗi lượt chỉ hỏi tối đa 1 câu và câu hỏi phải dựa trực tiếp vào câu trả lời gần nhất.
- Không hỏi lại điều người dùng đã nói; không hỏi máy móc kiểu “Bạn thích gì?” nhiều lần.
- Luân phiên kiểu câu hỏi: sở thích, lý do/động lực, tình huống thực tế, A/B, môi trường, cách xử lý thất bại, dự án muốn làm, giá trị/điều không muốn đánh đổi, cách học kỹ năng mới.
- Nếu người dùng đã nói rõ một lĩnh vực, đào sâu lĩnh vực đó trước khi mở rộng.
- Không kết luận quá sớm và không nói nghề nào là “định mệnh”.
- Trả lời 2-5 câu cho câu hỏi thông thường; chỉ dài hơn khi người dùng yêu cầu tổng hợp, phân tích hoặc gợi ý nghề.

QUY TẮC KHI NGƯỜI DÙNG MUỐN TỔNG HỢP / HỎI NGHỀ:
- Nếu người dùng hỏi kiểu “tổng hợp đi”, “thế nghề tui cần hướng là nghề gì”, “nghề nào hợp với tui”, “gợi ý nghề”, “nên theo nghề gì”, “định hướng nghề nghiệp”, hoặc tương tự: KHÔNG hỏi thêm câu nào trong lượt đó. Phải hoàn thành câu trả lời.
- Luôn dùng tiêu đề: “💼 NGHỀ NGHIỆP ĐÁNG THỬ”.
- Đưa 3-5 NGHỀ CỤ THỂ, không chỉ nêu tên ngành rộng.
- Mỗi nghề phải có đủ 4 ý ngắn: “Vì sao phù hợp”, “Dữ kiện từ cuộc trò chuyện”, “Điểm cần phát triển”, “Cách thử thực tế”.
- Sau danh sách nghề, luôn có “🧭 BƯỚC TIẾP THEO” với 2-4 việc nhỏ để người dùng kiểm chứng hướng đi.
- Không kết thúc giữa một nghề hoặc giữa một danh sách. Nếu cần ngắn gọn, hãy rút mỗi ý lại chứ không cắt mất nghề hoặc phần quan trọng.
- Không dùng các ví dụ nghề không có tín hiệu từ cuộc trò chuyện chỉ để làm đủ số lượng. Nếu dữ liệu thật sự chưa đủ, nói rõ “chưa đủ dữ liệu” và hỏi 1 câu trước ở các lượt thông thường.

CÁCH LẬP LUẬN:
- Ưu tiên các nghề giao thoa nhiều tín hiệu từ người dùng.
- Dữ kiện phải lấy từ những gì người dùng thực sự đã nói hoặc cách họ xử lý tình huống trong cuộc trò chuyện.
- Không chẩn đoán tính cách hay năng lực như sự thật tuyệt đối.
- Mọi gợi ý đều là giả thuyết để thử nghiệm, không phải kết luận.

ĐỊNH DẠNG:
- Ưu tiên tiêu đề rõ ràng, danh sách đánh số, bullet ngắn.
- Khi nêu nghề, dùng tên nghề cụ thể bằng tiếng Việt, có thể kèm tên tiếng Anh trong ngoặc.
`;


/* =========================================================
   TIỆN ÍCH PHÂN TÍCH HỘI THOẠI
   ========================================================= */

function normalizeForMatch(value) {
    return String(value || "")
        .toLowerCase()
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .replace(/[^\p{L}\p{N}]+/gu, " ")
        .replace(/\s+/g, " ")
        .trim();
}

function hasPhrase(text, phrase) {
    const haystack = ` ${normalizeForMatch(text)} `;
    const needle = ` ${normalizeForMatch(phrase)} `;
    return needle.length > 2 && haystack.includes(needle);
}

function isRecommendationRequest(message) {
    const t = normalizeForMatch(message);
    return [
        "tong hop", "tom lai", "ket luan", "goi y nghe", "goi y nghe nghiep",
        "nghe gi hop", "nghe nao hop", "nen theo nghe gi", "huong nghe gi",
        "huong nghiep", "dinh huong nghe nghiep", "nghe tui can huong",
        "nghe toi nen", "nghe phu hop", "3 nghe", "4 nghe", "5 nghe",
        "de xuat nghe", "chon nghe", "nen lam nghe gi", "cong viec phu hop"
    ].some(k => t.includes(k));
}

function extractCareerCandidates(text) {
    const source = String(text || "");
    const found = [];
    const add = (value) => {
        const clean = String(value || "")
            .replace(/^[\s#*`]+|[\s#*`]+$/g, "")
            .replace(/\s+/g, " ")
            .replace(/[.:：]+$/g, "")
            .trim();
        if (!clean || clean.length < 3 || clean.length > 90) return;
        const normalized = normalizeForMatch(clean);
        if (!normalized || found.some(x => normalizeForMatch(x) === normalized)) return;
        const banned = ["vi sao phu hop", "dieu kien", "cach thu", "diem can phat trien", "bước tiep theo", "buoc tiep theo"];
        if (banned.some(x => normalized === x || normalized.startsWith(`${x} `))) return;
        found.push(clean);
    };

    const numbered = /^\s*(?:\d+\s*[.)]|[-•])\s*(?:\*\*)?([^\n:]{3,100}?)(?:\*\*)?\s*(?:[-–—:]|$)/gmi;
    let match;
    while ((match = numbered.exec(source)) && found.length < 6) add(match[1]);

    if (found.length < 3) {
        const boldLine = /^\s*\*\*([^*\n]{3,90})\*\*\s*$/gmi;
        while ((match = boldLine.exec(source)) && found.length < 6) add(match[1]);
    }

    return found.slice(0, 6);
}

function isCareerRecommendation(text) {
    const source = String(text || "");
    return /NGHỀ NGHIỆP ĐÁNG THỬ|CAREERS WORTH TRYING/i.test(source) || extractCareerCandidates(source).length >= 3;
}

function getConversationFingerprint(session) {
    const payload = (session?.messages || [])
        .slice(-40)
        .map(m => `${m.role}:${m.text}`)
        .join("\n");
    return crypto.createHash("sha256").update(payload, "utf8").digest("hex");
}

function getTranscript(session, limit = 28) {
    return (session?.messages || [])
        .slice(-limit)
        .map((m) => `${m.role === "user" ? "NGƯỜI DÙNG" : "AI"}: ${String(m.text || "").slice(0, 1800)}`)
        .join("\n\n");
}

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
                profileCache: null,
                personalizedQuizCache: new Map(),
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
            activeGeminiRequests,
            queuedGeminiRequests: geminiQueue.length,
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
        interests: {
            type: "array",
            minItems: 0,
            maxItems: 6,
            items: { type: "string" }
        },
        strengths: {
            type: "array",
            minItems: 0,
            maxItems: 6,
            items: { type: "string" }
        },
        thinking_style: { type: "string" },
        motivations: { type: "string" },
        work_environment: { type: "string" },
        growth_areas: {
            type: "array",
            minItems: 0,
            maxItems: 5,
            items: { type: "string" }
        },
        career_directions: {
            type: "array",
            minItems: 0,
            maxItems: 5,
            items: {
                type: "object",
                properties: {
                    career: { type: "string" },
                    why: { type: "string" },
                    evidence: {
                        type: "array",
                        minItems: 0,
                        maxItems: 4,
                        items: { type: "string" }
                    },
                    develop: { type: "string" },
                    experiment: { type: "string" }
                },
                required: ["career", "why", "evidence", "develop", "experiment"],
                additionalProperties: false
            }
        }
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
    profileLimiter,
    async (req, res) => {
        const sessionId = req.headers["x-session-id"];

        if (typeof sessionId !== "string" || !sessions.has(sessionId)) {
            return res.status(400).json({
                error: "Chưa có phiên trò chuyện. Hãy nhắn với AI trước."
            });
        }

        const session = sessions.get(sessionId);
        session.lastUsedAt = Date.now();
        const fingerprint = getConversationFingerprint(session);

        if (!Array.isArray(session.messages) || !session.messages.some(m => m.role === "user")) {
            return res.status(400).json({
                error: "Hãy nhắn với AI ít nhất một lần để hồ sơ có dữ liệu."
            });
        }

        if (session.profileCache?.fingerprint === fingerprint && session.profileCache?.profile) {
            return res.json({
                ok: true,
                cached: true,
                profile: session.profileCache.profile
            });
        }

        try {
            const transcript = getTranscript(session, 30);
            const prompt = `
Hãy tạo HỒ SƠ HƯỚNG NGHIỆP RÕ RÀNG cho người dùng dựa trên TOÀN BỘ cuộc trò chuyện dưới đây.

Mục tiêu của hồ sơ:
- Tóm tắt các tín hiệu đã được người dùng thể hiện.
- Đưa ra 3-5 hướng nghề CỤ THỂ đáng khám phá khi dữ liệu đủ.
- Mỗi hướng nghề phải giải thích rõ vì sao phù hợp, bằng chứng nào trong cuộc trò chuyện dẫn tới gợi ý đó, điều gì cần phát triển và một cách thử thực tế.

Nguyên tắc:
- Chỉ dùng dữ kiện thật sự xuất hiện trong transcript. Không bịa.
- Có thể suy luận ở mức “giả thuyết nghề nghiệp” nếu có nhiều tín hiệu hỗ trợ, nhưng phải diễn đạt như một hướng để thử nghiệm, không phải kết luận chắc chắn.
- Không gắn nhãn cố định, không chẩn đoán tính cách, không khẳng định người dùng chắc chắn phù hợp với một nghề.
- Nếu một mục chưa có đủ dữ liệu, ghi ngắn gọn “Chưa đủ dữ liệu” thay vì bịa.
- career_directions phải là danh sách đối tượng có các trường: career, why, evidence, develop, experiment.
- Ưu tiên nghề cụ thể, ví dụ chức danh công việc, thay vì chỉ nói “CNTT”, “kinh doanh”, “y tế”.
- Nếu dữ liệu đủ để gợi ý nghề, hãy cố gắng trả 3-5 hướng nghề có chất lượng thay vì trả một danh sách tên nghề không giải thích.
- Trả JSON đúng schema, không markdown, không văn bản ngoài JSON.

LỊCH SỬ CUỘC TRÒ CHUYỆN:
${transcript}
`;

            const profileResponse = await createGenerateContentWithRetry({
                model: MODEL,
                contents: prompt,
                config: {
                    responseMimeType: "application/json",
                    responseSchema: PROFILE_SCHEMA,
                    systemInstruction:
                        "Bạn là AI phân tích hồ sơ hướng nghiệp. Trả JSON hợp lệ theo schema. Chỉ dùng dữ kiện trong transcript; mọi gợi ý nghề đều là giả thuyết để khám phá."
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

            if (!profile || !Array.isArray(profile.interests) || !Array.isArray(profile.strengths) ||
                typeof profile.thinking_style !== "string" || typeof profile.motivations !== "string" ||
                typeof profile.work_environment !== "string" || !Array.isArray(profile.growth_areas) ||
                !Array.isArray(profile.career_directions)) {
                throw new Error("Dữ liệu hồ sơ không đúng định dạng.");
            }

            profile.interests = profile.interests.filter(Boolean).map(String).slice(0, 6);
            profile.strengths = profile.strengths.filter(Boolean).map(String).slice(0, 6);
            profile.growth_areas = profile.growth_areas.filter(Boolean).map(String).slice(0, 5);
            profile.career_directions = profile.career_directions.filter(Boolean).map(item => ({
                career: String(item.career || "").trim(),
                why: String(item.why || "").trim(),
                evidence: Array.isArray(item.evidence) ? item.evidence.filter(Boolean).map(String).slice(0, 4) : [],
                develop: String(item.develop || "").trim(),
                experiment: String(item.experiment || "").trim()
            })).filter(item => item.career).slice(0, 5);

            session.profileCache = { fingerprint, profile, createdAt: Date.now() };
            return res.json({ ok: true, cached: false, profile });

        } catch (error) {
            console.error("Profile error:", error);
            return res.status(503).json({
                error: getFriendlyError(error),
                retryable: isRetryableGeminiError(error)
            });
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
        const message = typeof req.body?.message === "string" ? req.body.message.trim() : "";

        if (!message) {
            return res.status(400).json({ error: "Tin nhắn không được để trống." });
        }

        if (message.length > 2000) {
            return res.status(400).json({ error: "Tin nhắn quá dài. Vui lòng nhập tối đa 2000 ký tự." });
        }

        const session = getSession(req, res);
        session.messages ||= [];
        session.messages.push({ role: "user", text: message, at: Date.now() });
        if (session.messages.length > 40) session.messages = session.messages.slice(-40);
        // Nội dung hồ sơ phải được tính lại sau một tin nhắn mới.
        session.profileCache = null;

        let releaseGeminiSlot;
        try {
            releaseGeminiSlot = await acquireGeminiSlot();
        } catch (error) {
            session.messages.pop();
            return res.status(503).json({ error: getFriendlyError(error) });
        }

        res.status(200);
        res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
        res.setHeader("Cache-Control", "no-cache, no-transform");
        res.setHeader("Connection", "keep-alive");
        res.setHeader("X-Accel-Buffering", "no");
        if (typeof res.flushHeaders === "function") res.flushHeaders();

        const sendEvent = payload => {
            if (!res.writableEnded) res.write(`data: ${JSON.stringify(payload)}\n\n`);
        };

        let fullText = "";
        let streamStarted = false;

        try {
            const wantsRecommendation = isRecommendationRequest(message);
            // Không duplicate tin nhắn mới nhất trong transcript.
            const priorTranscript = getTranscript({ messages: (session.messages || []).slice(0, -1) }, 20);

            const chatPrompt = `
${SYSTEM_INSTRUCTION}

LƯỢT HIỆN TẠI:
${wantsRecommendation
    ? "Người dùng đang yêu cầu tổng hợp hoặc định hướng nghề. Hãy hoàn thành trọn vẹn câu trả lời ngay; TUYỆT ĐỐI không hỏi thêm câu ở cuối lượt này."
    : "Đây là một lượt trò chuyện khám phá. Nếu còn thiếu dữ liệu, tối đa 1 câu hỏi tiếp theo."}

LỊCH SỬ GẦN ĐÂY:
${priorTranscript || "(Chưa có lịch sử trước đó.)"}

TIN NHẮN MỚI NHẤT CỦA NGƯỜI DÙNG:
${message}

${wantsRecommendation ? `
KIỂM TRA TRƯỚC KHI TRẢ LỜI:
- Phải có tiêu đề “💼 NGHỀ NGHIỆP ĐÁNG THỬ”.
- Phải có 3-5 nghề cụ thể và hoàn thành đủ 4 ý cho từng nghề.
- Phải có “🧭 BƯỚC TIẾP THEO”.
- Không kết thúc giữa danh sách, không bỏ dở nghề cuối cùng.
` : ""}
`;

            const stream = await createGenerateContentStreamWithRetry({
                model: MODEL,
                contents: chatPrompt,
                config: {
                    systemInstruction: "Bạn là AI hướng nghiệp thân thiện, thực tế và luôn trả lời bằng tiếng Việt. Hãy ưu tiên hoàn thành yêu cầu của người dùng, không cắt dở danh sách.",
                    thinkingConfig: { thinkingLevel: "low" },
                    maxOutputTokens: wantsRecommendation ? 1500 : 720
                }
            }, 4);

            streamStarted = true;
            for await (const chunk of stream) {
                const chunkText = typeof chunk?.text === "string" ? chunk.text : "";
                if (!chunkText) continue;
                fullText += chunkText;
                sendEvent({ type: "text", text: chunkText });
            }

            if (!fullText.trim()) throw new Error("Gemini không trả về nội dung.");

            session.lastAssistantText = fullText;
            session.messages.push({ role: "assistant", text: fullText, at: Date.now() });
            if (session.messages.length > 40) session.messages = session.messages.slice(-40);

            if (isCareerRecommendation(fullText)) {
                session.lastRecommendation = {
                    interactionId: null,
                    text: fullText,
                    createdAt: Date.now()
                };
            }

            sendEvent({ type: "done" });
            res.end();

        } catch (error) {
            console.error("Gemini error:", error);

            // Nếu Gemini fail trước khi có nội dung, không để một lượt lỗi làm bẩn hồ sơ.
            if (!fullText.trim()) {
                const last = session.messages?.[session.messages.length - 1];
                if (last?.role === "user" && last.text === message) session.messages.pop();
            } else if (streamStarted) {
                session.lastAssistantText = fullText;
            }

            sendEvent({ type: "error", message: getFriendlyError(error), retryable: isRetryableGeminiError(error) });
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
    return Object.entries(INDUSTRY_RULES)
        .filter(([, rule]) => rule.terms.some(term => hasPhrase(text, term)))
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
        session.lastUsedAt = Date.now();
        const recommendation = session.lastRecommendation;
        const recommendationText = recommendation?.text || "";

        if (!isCareerRecommendation(recommendationText)) {
            return res.status(403).json({ error: "Quiz riêng chỉ mở sau khi AI đưa ra gợi ý nghề nghiệp." });
        }

        const matches = getMatchingIndustries(recommendationText);
        if (!matches.includes(industry)) {
            return res.status(403).json({ error: "Nghề được gợi ý chưa có bộ quiz riêng cho lĩnh vực này." });
        }

        const fingerprint = getConversationFingerprint(session);
        const cacheKey = `${fingerprint}:${industry}`;
        const cached = session.personalizedQuizCache?.get(cacheKey);
        if (cached) return res.json({ ...cached, cached: true });

        let releaseGeminiSlot;
        try {
            releaseGeminiSlot = await acquireGeminiSlot();

            const transcript = getTranscript(session, 34);
            const prompt = `
Hãy tạo QUIZ HƯỚNG NGHIỆP RIÊNG gồm ĐÚNG 20 câu cho người dùng hiện tại.

Lĩnh vực được gợi ý: ${INDUSTRY_RULES[industry].title}
Nghề/lĩnh vực hiển thị trên nút: ${careerLabel || INDUSTRY_RULES[industry].title}

Lịch sử cuộc trò chuyện:
${transcript}

Phần gợi ý nghề gần nhất của AI:
${recommendationText}

YÊU CẦU:
- Dựa vào TOÀN BỘ ngữ cảnh trên để cá nhân hóa cách đặt tình huống.
- Không hỏi lại nguyên văn câu trong cuộc trò chuyện.
- Không biến quiz thành bài kiểm tra kiến thức chuyên ngành nặng.
- 20 câu, mỗi câu đúng 4 lựa chọn.
- answer là chỉ số 0,1,2,3 của đáp án đúng.
- explanation ngắn, giải thích logic của tình huống và giữ giọng trung tính.
- Phân bố vị trí đáp án đúng đa dạng.
- Các câu phải giúp phân biệt sở thích, cách ra quyết định, cách giải quyết vấn đề, độ kiên nhẫn, môi trường làm việc và động lực liên quan tới lĩnh vực này.
- Không kết luận “chắc chắn hợp nghề”; chỉ phục vụ khám phá.
- Tiếng Việt, JSON thuần theo schema.
`;

            const response = await createGenerateContentWithRetry({
                model: MODEL,
                contents: prompt,
                config: {
                    responseMimeType: "application/json",
                    responseSchema: PERSONAL_QUIZ_SCHEMA,
                    systemInstruction: "Bạn là AI thiết kế quiz hướng nghiệp cá nhân hóa. Trả đúng 20 câu JSON hợp lệ và dựa vào transcript được cung cấp."
                }
            }, 4);

            const outputText = typeof response?.text === "string" ? response.text.trim() : "";
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
                question: String(q?.question || `Câu ${index + 1}`).trim(),
                options: Array.isArray(q?.options) ? q.options.slice(0, 4).map(x => String(x).trim()) : [],
                answer: Number(q?.answer),
                explanation: String(q?.explanation || "").trim()
            }));

            if (questions.some(q => !q.question || q.options.length !== 4 || q.options.some(x => !x) ||
                !Number.isInteger(q.answer) || q.answer < 0 || q.answer > 3 || !q.explanation)) {
                return res.status(502).json({ error: "AI trả về một câu hỏi không hợp lệ. Vui lòng thử lại." });
            }

            const payload = {
                ok: true,
                title: `Quiz riêng: ${INDUSTRY_RULES[industry].title}`,
                questions
            };
            session.personalizedQuizCache ||= new Map();
            session.personalizedQuizCache.set(cacheKey, payload);
            return res.json({ ...payload, cached: false });

        } catch (error) {
            console.error("Personalized quiz error:", error);
            return res.status(503).json({ error: getFriendlyError(error), retryable: isRetryableGeminiError(error) });
        } finally {
            releaseGeminiSlot?.();
        }
    }
);


/* =========================================================
   KHÔI PHỤC PHIÊN
   ========================================================= */

app.get(
    "/api/session",
    (req, res) => {
        const sessionId = req.headers["x-session-id"];
        if (typeof sessionId !== "string" || !sessions.has(sessionId)) {
            return res.status(404).json({ error: "Phiên không còn tồn tại." });
        }
        const session = sessions.get(sessionId);
        session.lastUsedAt = Date.now();
        return res.json({
            ok: true,
            messages: (session.messages || []).slice(-40)
        });
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
    const message = getErrorText(error) || "Lỗi không xác định.";

    if (error?.code === "SERVER_BUSY" || message === "SERVER_BUSY") {
        return "Máy chủ đang có nhiều người dùng cùng lúc. Vui lòng thử lại sau ít giây.";
    }
    if (/API key|API_KEY/i.test(message)) {
        return "Kết nối AI của server đang có vấn đề. Vui lòng thử lại sau hoặc kiểm tra cấu hình server.";
    }
    if (/429|rate.?limit|resource.?exhausted/i.test(message)) {
        return "AI đang nhận quá nhiều yêu cầu cùng lúc. Mình đã tự thử lại nhưng vẫn chưa thành công. Bạn bấm “Thử lại” sau vài giây nhé.";
    }
    if (/503|service.?unavailable|unavailable|high demand|overloaded/i.test(message)) {
        return "AI đang hơi quá tải một chút. Hệ thống đã tự thử kết nối lại; bạn thử lại sau vài giây nhé.";
    }
    if (/500|502|504|internal|temporar|deadline.?exceeded|timeout|econnreset|socket/i.test(message)) {
        return "Kết nối AI đang gặp lỗi tạm thời. Bạn thử lại sau vài giây nhé.";
    }
    if (/403/.test(message)) {
        return "Gemini API từ chối quyền truy cập. Hãy kiểm tra API key và project Google.";
    }
    if (/404/.test(message)) {
        return "Mô hình AI tạm thời không sẵn sàng. Vui lòng thử lại sau ít giây.";
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
