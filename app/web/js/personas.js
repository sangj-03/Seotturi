// Who the user talks to in AI 코칭: interviewers, audiences, and for daily conversation one cast per
// situation (a cafe order is taken by cafe staff, not by a friend). Every persona has
//   name   unique across modes; saved sessions and /api/tts refer to it
//   short  how the chat names them while they answer ("하은이 대답하는 중")
//   desc   the selection card text (also given to the feedback prompt)
//   act    daily only: how Gemma plays them (server/analysis.py reply_messages)
//   opener daily only: their first line; hello: interview/presentation, said before the question
//   voice  [Supertonic 3 preset, speed] for server/tts.py. Female presets by measured median F0:
//          F2 258 Hz, F1 209, F4 193 (flattest), F3 182, F5 158; male: M1 176 (liveliest), M4 136,
//          M5 116, M3 111, M2 93
//   img    the scene behind the session (portrait, the character centred, face in the upper half);
//          without one the mode's scene is shown. face: the avatar on the selection card.

const P = "/assets/personas/";

export const INTERVIEWERS = [
  { id: "warm-mentor", name: "따뜻한 멘토형", short: "멘토", voice: ["F1", 1.0],
    desc: "미소를 띠고 끝까지 들어 주는 면접관이에요. 긴장을 풀고 차분하게 답하는 연습에 좋아요.",
    hello: "편하게 말씀해 주세요." },
  { id: "strict-manager", name: "깐깐한 부장님형", short: "부장님", voice: ["M5", 0.95], img: "/assets/interviewer.jpg",
    desc: "날카로운 눈빛의 부장님이에요. 꼬리 질문에도 논리와 책임감을 보여 주는 연습을 해요.",
    hello: "핵심만 말씀하세요." },
  { id: "calm-hr", name: "차분한 인사팀장형", short: "인사팀장", voice: ["M3", 0.9], legacy: ["차분한 교수형"],
    desc: "안경을 쓴 침착한 인사팀장이에요. 생각을 정리해 차근차근 답하는 사고력을 봐요.",
    hello: "천천히 생각을 정리해서 말씀해 보세요." },
  { id: "startup-leader", name: "유쾌한 스타트업 리더형", short: "대표", voice: ["M1", 1.12],
    desc: "캐주얼한 차림의 젊은 대표예요. 창의성과 도전 경험을 묻는 편안한 분위기에서 연습해요.",
    hello: "반가워요! 편하게 얘기해 봐요." },
  { id: "cool-exec", name: "냉철한 임원형", short: "임원", voice: ["F4", 1.05],
    desc: "결론과 근거를 중시하는 임원이에요. 숫자와 사례로 짧고 분명하게 답하는 연습에 좋아요.",
    hello: "근거와 수치 위주로 답변해 주세요." },
];

export const AUDIENCES = [
  { id: "friendly-team", name: "우호적인 동료들", short: "동료들", voice: ["F3", 1.05], img: "/assets/audience.jpg",
    desc: "고개를 끄덕이며 들어 주는 팀 동료들이에요. 편안한 분위기에서 전달력과 흐름을 연습해요.",
    hello: "준비되면 시작해 주세요!" },
  { id: "sharp-judges", name: "날카로운 심사위원", short: "심사위원", voice: ["M2", 1.0],
    desc: "공모전·경진대회 심사위원단이에요. 핵심 메시지와 근거, 질의응답 대응을 꼼꼼히 봐요.",
    hello: "발표 시작하세요." },
  { id: "busy-execs", name: "바쁜 임원진", short: "임원진", voice: ["F5", 1.12],
    desc: "결론부터 듣고 싶어 하는 임원 회의예요. 두괄식으로 짧고 간결하게 말하는 연습을 해요.",
    hello: "결론부터 말씀해 주시죠." },
];

const JIMIN = { id: "jimin", name: "다정한 친구 지민", short: "지민", voice: ["F2", 1.05],
  desc: "잘 웃고 맞장구를 잘 쳐 주는 친구예요. 편하게 수다 떨며 표준어 감각을 익혀요.",
  act: "잘 웃고 맞장구를 잘 쳐 주는 친구다. 친한 친구에게 하듯 다정한 반말로 말한다.",
  opener: "야, 오랜만이다! 요즘 어떻게 지내?" };
const DOYUN = { id: "doyun", name: "장난꾸러기 친구 도윤", short: "도윤", voice: ["M4", 1.12], img: "/assets/friend.jpg",
  desc: "농담을 좋아하는 유쾌한 친구예요. 빠른 대화 속에서도 표준어를 유지해 봐요.",
  act: "농담과 장난을 좋아하고 말이 빠른 친구다. 유쾌한 반말로 말한다.",
  opener: "오, 왔어? 나 오늘 진짜 웃긴 일 있었는데 들어 볼래?" };
const SEOJIN = { id: "seojin", name: "무뚝뚝한 친구 서진", short: "서진", voice: ["M2", 0.95],
  desc: "말수는 적지만 진지한 친구예요. 또박또박 말하는 연습에 좋아요.",
  act: "말수가 적고 무뚝뚝하지만 진지하게 들어 주는 친구다. 반말로 짧게 말한다.",
  opener: "왔네. 밥은 먹었어?" };

// Daily conversation: the situations of the setup step, in order, each with its own cast.
export const SITUATIONS = {
  "친구와 수다": { role: "친구", pick: "함께 수다 떨 친구를 선택해 주세요", cast: [JIMIN, DOYUN, SEOJIN] },
  "카페에서 주문": { role: "카페 직원", pick: "주문을 받을 카페 직원을 선택해 주세요", cast: [
    { id: "barista", name: "친절한 바리스타 하은", short: "하은", voice: ["F1", 1.0],
      desc: "천천히 기다려 주는 친절한 직원이에요. 메뉴와 옵션을 차근차근 주문하는 연습을 해요.",
      act: "밝고 친절한 카페 직원이다. 손님이 천천히 말해도 기다려 주고, 메뉴를 추천해 준다. 존댓말을 쓴다.",
      opener: "어서 오세요! 천천히 보시고 주문하실 때 말씀해 주세요." },
    { id: "rush-staff", name: "바쁜 알바생 준호", short: "준호", voice: ["M1", 1.15],
      desc: "손님이 몰린 점심시간, 말이 빠른 직원이에요. 한 번에 또박또박 주문하는 연습에 좋아요.",
      act: "점심시간이라 손님이 많아 바쁜 카페 직원이다. 말이 빠르고 짧게 묻는다. 메뉴, 사이즈, 매장 또는 포장을 빠르게 확인한다. 존댓말을 쓴다.",
      opener: "다음 손님, 주문하시겠어요? 메뉴 정하셨으면 바로 말씀해 주세요!" },
    { id: "cafe-owner", name: "꼼꼼한 카페 사장님", short: "사장님", voice: ["F5", 1.0],
      desc: "옵션을 하나하나 물어보는 동네 카페 사장님이에요. 사이즈, 샷 추가, 포장 여부까지 길게 말하는 연습을 해요.",
      act: "동네 카페 사장님이다. 친절하지만 꼼꼼해서 사이즈, 온도, 샷 추가, 우유 변경, 매장 또는 포장, 포인트 적립을 차례로 묻는다. 존댓말을 쓴다.",
      opener: "어서 오세요. 오늘은 어떤 음료로 준비해 드릴까요?" },
  ] },
  "길 묻기": { role: "길에서 만난 행인", pick: "길을 물어볼 사람을 선택해 주세요", cast: [
    { id: "student", name: "친절한 대학생", short: "대학생", voice: ["F2", 1.05],
      desc: "휴대폰 지도를 보여 주며 친절하게 알려 주는 대학생이에요. 가려는 곳을 정확히 묻는 연습을 해요.",
      act: "길에서 만난 친절한 대학생이다. 휴대폰 지도를 보며 방향과 걸리는 시간을 자세히 알려 준다. 존댓말을 쓴다.",
      opener: "네? 길 찾으세요? 어디 가시는데요?" },
    { id: "office-worker", name: "바쁜 직장인", short: "직장인", voice: ["M5", 1.12],
      desc: "시계를 보며 서두르는 출근길 직장인이에요. 짧고 정확하게 묻고 핵심을 알아듣는 연습을 해요.",
      act: "출근길이라 바쁜 직장인이다. 시계를 보며 짧게 대답하고, 질문이 길면 핵심을 다시 묻는다. 존댓말을 쓴다.",
      opener: "아, 네. 제가 좀 급해서요. 어디 찾으세요?" },
    { id: "grandma", name: "동네 토박이 할머니", short: "할머니", voice: ["F5", 0.88],
      desc: "동네를 훤히 아는 인자한 할머니예요. 천천히 또렷하게 묻고 정중하게 말하는 연습에 좋아요.",
      act: "이 동네에 오래 산 인자한 할머니다. 길을 천천히 아주 자세히 설명하고, 잘 못 알아들으면 다시 물어본다. 다정한 해요체로 말한다.",
      opener: "아이고, 길 찾아요? 어디 가는데요?" },
  ] },
  "직장 동료와 대화": { role: "직장 동료", pick: "대화할 직장 동료를 선택해 주세요", cast: [
    { id: "peer", name: "입사 동기 수아", short: "수아", voice: ["F3", 1.05],
      desc: "같은 날 입사한 동기예요. 일상과 업무 이야기를 편하게 나누며 표준어를 다듬어요.",
      act: "같은 날 입사한 친한 동기다. 편한 반말로 일상과 업무 이야기를 나눈다.",
      opener: "주말 잘 보냈어? 나는 집에서 푹 쉬었어." },
    { id: "team-lead", name: "꼼꼼한 팀장님", short: "팀장님", voice: ["M3", 0.95],
      desc: "업무를 꼼꼼히 챙기는 팀장님이에요. 보고하듯 정중한 존댓말로 또박또박 말하는 연습을 해요.",
      act: "꼼꼼하고 진지하지만 따뜻한 팀장이다. 업무 진행 상황을 묻고 핵심을 다시 확인한다. 존댓말을 쓴다.",
      opener: "주말은 잘 보냈어요? 회의 전에 진행 상황 좀 들어 볼까요?" },
    { id: "junior", name: "신입 후배 민재", short: "민재", voice: ["M1", 1.05],
      desc: "궁금한 게 많은 의욕 넘치는 신입이에요. 업무를 쉽고 분명하게 설명하는 연습을 해요.",
      act: "입사한 지 일주일 된 의욕 넘치는 신입이다. 선배에게 존댓말로 업무를 이것저것 묻는다.",
      opener: "선배님, 안녕하세요! 혹시 지금 잠깐 여쭤봐도 될까요?" },
  ] },
  "자유 주제": { role: "친구", pick: "자유롭게 이야기할 상대를 선택해 주세요", cast: [
    { id: "turi", name: "말동무 서투리", short: "서투리", voice: ["M4", 1.05], role: "말동무",
      desc: "서투리 마스코트예요. 어떤 주제든 함께 이야기하며 표준어로 말하는 감각을 키워요.",
      act: "서투리 앱의 마스코트이자 다정한 말동무다. 어떤 주제든 호기심을 갖고 존댓말로 즐겁게 이야기한다.",
      opener: "안녕하세요, 저는 서투리예요! 오늘은 무슨 이야기를 해 볼까요?" },
    { ...JIMIN, opener: "안녕! 오늘은 무슨 얘기 해 볼까?" },
    { ...DOYUN, opener: "심심했는데 잘 왔다! 오늘은 무슨 얘기 할래?" },
    { ...SEOJIN, opener: "왔네. 오늘은 무슨 얘기 할까?" },
  ] },
};

// The scenes and avatars generated for the personas (web/assets/personas/<id>.jpg, <id>.face.jpg).
const DRAWN = new Set(["warm-mentor", "calm-hr", "startup-leader", "cool-exec", "sharp-judges", "busy-execs", "jimin", "seojin",
  "barista", "rush-staff", "cafe-owner", "student", "office-worker", "grandma", "peer", "team-lead", "junior", "turi"]);
const FACES = new Set([...DRAWN, "strict-manager", "doyun", "friendly-team"]);
for (const p of [...INTERVIEWERS, ...AUDIENCES, ...Object.values(SITUATIONS).flatMap((s) => s.cast)]) {
  if (!p.img && DRAWN.has(p.id)) p.img = `${P}${p.id}.jpg`;
  if (FACES.has(p.id)) p.face = `${P}${p.id}.face.jpg`;
}

const ALL = [...INTERVIEWERS, ...AUDIENCES, ...Object.values(SITUATIONS).flatMap((s) => s.cast)];

/** The cast offered for a project: interviewers, audiences, or the daily situation's partners. */
export function castOf(mode, category) {
  if (mode === "interview") return INTERVIEWERS;
  if (mode === "presentation") return AUDIENCES;
  return (SITUATIONS[category] || SITUATIONS["친구와 수다"]).cast;
}

/** A persona by its name, also under a name it had before (sessions saved then keep their voice). */
export function byName(name) {
  return ALL.find((p) => p.name === name || (p.legacy || []).includes(name)) || null;
}

/** "지민" -> "지민이", "준호" -> "준호가": the subject particle after a Korean word. */
export function subject(word) {
  const c = word.charCodeAt(word.length - 1);
  const batchim = c >= 0xac00 && c <= 0xd7a3 && (c - 0xac00) % 28 !== 0;
  return word + (batchim ? "이" : "가");
}
