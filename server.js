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

const MODEL =
    "gemini-3.8-flash";

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

const chatLimiter =
    rateLimit({

        windowMs:
            10 * 60 * 1000,

        limit: 30,

        standardHeaders: true,

        legacyHeaders: false,

        message: {
            error:
                "Bạn gửi quá nhiều tin nhắn. Vui lòng thử lại sau ít phút."
        }

    });


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
                await ai.interactions.create(
                    request
                );


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

                        type:
                            "error",

                        message:
                            errorMessage

                    });

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
            "Không tìm thấy model Gemini. " +
            "Hãy kiểm tra model trong server.js."
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
