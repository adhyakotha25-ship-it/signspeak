/**
 * SignSpeak — Word Detection Engine
 * ─────────────────────────────────────────────────────────────
 * Uses MediaPipe Holistic for full-body landmark detection and an embedded
 * decision-tree / heuristic classifier for static ASL letters.
 * All processing is done locally in the browser.
 */

'use strict';

/* ============================================================
   Constants & Configuration
   ============================================================ */
const CONFIG = {
  /** Minimum MediaPipe confidence to consider a detection valid */
  MIN_DETECTION_CONFIDENCE: 0.7,
  MIN_TRACKING_CONFIDENCE:  0.6,
  MIN_FACE_CONFIDENCE:      0.55,
  MIN_POSE_CONFIDENCE:      0.55,
  /** High-confidence gate for accepting any frame prediction */
  MIN_WORD_CONFIDENCE: 0.70,
  /** Reject ambiguous frames where top classes are too close */
  MIN_SCORE_MARGIN: 0.13,
  /** Rolling frame buffer for temporal smoothing (smaller = snappier) */
  PREDICTION_WINDOW_SIZE: 7,
  /** Majority threshold: this fraction of buffered frames must agree */
  MAJORITY_RATIO: 0.55,
  /** Pose must stay stable for this long before commit */
  GESTURE_HOLD_MS: 240,
  /** Ignore predictions when average keypoint velocity is high.
   *  Raised so genuine wave signs (hello/goodbye) can still buffer. */
  MOTION_VELOCITY_THRESHOLD: 4.2,
  /** Cooldown after commit to prevent double-firing */
  COOLDOWN_MS: 750,
  /** Canvas rendering */
  LANDMARK_COLOR:  '#6D28D9',
  CONNECTOR_COLOR: 'rgba(109,40,217,0.5)',
  BBOX_COLOR:      'rgba(6,182,212,0.8)',
  BBOX_LINE_WIDTH: 2,
  BBOX_CORNER_LEN: 16,
};

const WORD_DESCRIPTIONS = {
  hello: 'Open hand greeting motion',
  goodbye: 'Waving hand side to side',
  yes: 'Fist nodding motion',
  no: 'Index and middle tapping thumb',
  please: 'Flat hand circling on chest',
  'thank you': 'Fingers from chin moving forward',
  sorry: 'Fist circling over chest',
  help: 'Thumb-up hand lifted from palm',
  eat: 'Pinched fingers tapping mouth',
  drink: 'C-hand tipping to mouth',
  mother: 'Open hand with thumb at chin area',
  father: 'Open hand with thumb at forehead area',
  stop: 'Flat hand chopping onto palm',
  want: 'Both open palms up, pull toward chest',
  friend: 'Hooked index fingers linked',
  work: 'Two fists tap together',
  school: 'Flat palms clap lightly',
  family: 'F-shaped hands circle together',
};

const DETECTABLE_WORDS = new Set([
  'hello',
  'goodbye',
  'yes',
  'no',
  'please',
  'thank you',
  'sorry',
  'help',
  'eat',
  'drink',
  'mother',
  'father',
  'stop',
  'want',
  'friend',
  'work',
  'school',
  'family',
]);

const WORD_MIN_CONFIDENCE = {
  'thank you': 0.80,
  mother: 0.84,
  father: 0.82,
  sorry: 0.86,
  please: 0.84,
  hello: 0.82,
  goodbye: 0.80,
  yes: 0.82,
  no: 0.80,
  eat: 0.82,
  drink: 0.78,
  stop: 0.80,
  want: 0.78,
  friend: 0.78,
  work: 0.76,
  school: 0.76,
  family: 0.76,
};

const BSL_COMMON_WORDS = [
  { word: 'hello', category: 'greetings', tip: 'Open hand near forehead and move out.' },
  { word: 'goodbye', category: 'greetings', tip: 'Wave hand side to side.' },
  { word: 'please', category: 'manners', tip: 'Flat hand circles on chest.' },
  { word: 'thank you', category: 'manners', tip: 'Fingers from chin move forward.' },
  { word: 'sorry', category: 'manners', tip: 'Fist circles on chest.' },
  { word: 'yes', category: 'manners', tip: 'Nod fist up and down.' },
  { word: 'no', category: 'manners', tip: 'Index and middle tap thumb.' },
  { word: 'help', category: 'needs', tip: 'Thumb-up hand lifts from palm.' },
  { word: 'want', category: 'needs', tip: 'Pull both hands toward you.' },
  { word: 'need', category: 'needs', tip: 'Hooked finger pulls down firmly.' },
  { word: 'eat', category: 'daily life', tip: 'Pinched fingers tap mouth.' },
  { word: 'drink', category: 'daily life', tip: 'C hand tips to mouth.' },
  { word: 'toilet', category: 'daily life', tip: 'Letter T handshake movement.' },
  { word: 'sleep', category: 'daily life', tip: 'Hand closes from open near face.' },
  { word: 'work', category: 'daily life', tip: 'Fists tap each other.' },
  { word: 'school', category: 'daily life', tip: 'Flat palms clap lightly.' },
  { word: 'home', category: 'daily life', tip: 'Move hand from mouth to cheek.' },
  { word: 'shop', category: 'daily life', tip: 'Flick finger from palm outward.' },
  { word: 'money', category: 'daily life', tip: 'Tap fingers on opposite palm.' },
  { word: 'phone', category: 'daily life', tip: 'Thumb and pinky by ear.' },
  { word: 'name', category: 'people', tip: 'Two fingers of each hand tap.' },
  { word: 'my', category: 'people', tip: 'Flat hand touches chest.' },
  { word: 'your', category: 'people', tip: 'Flat hand points outward.' },
  { word: 'me', category: 'people', tip: 'Point to your chest.' },
  { word: 'you', category: 'people', tip: 'Point toward person.' },
  { word: 'we', category: 'people', tip: 'Index circles group area.' },
  { word: 'they', category: 'people', tip: 'Point to side group.' },
  { word: 'friend', category: 'people', tip: 'Hooked index fingers link twice.' },
  { word: 'family', category: 'people', tip: 'F hands circle together.' },
  { word: 'baby', category: 'people', tip: 'Cradle arms like holding baby.' },
  { word: 'mother', category: 'people', tip: 'Thumb on chin.' },
  { word: 'father', category: 'people', tip: 'Thumb on forehead.' },
  { word: 'sister', category: 'people', tip: 'L hand from chin to side.' },
  { word: 'brother', category: 'people', tip: 'L hand from forehead to side.' },
  { word: 'woman', category: 'people', tip: 'Thumb traces jawline.' },
  { word: 'man', category: 'people', tip: 'Thumb on forehead.' },
  { word: 'child', category: 'people', tip: 'Flat hand pats low height.' },
  { word: 'teacher', category: 'people', tip: 'Teach sign: hands move from forehead out.' },
  { word: 'doctor', category: 'people', tip: 'Two fingers tap wrist pulse.' },
  { word: 'police', category: 'people', tip: 'Flat hand taps shoulder badge.' },
  { word: 'where', category: 'questions', tip: 'Index wiggles side to side.' },
  { word: 'what', category: 'questions', tip: 'Both flat hands turn up.' },
  { word: 'when', category: 'questions', tip: 'Index circles then points wrist.' },
  { word: 'who', category: 'questions', tip: 'Index finger circles at chin.' },
  { word: 'why', category: 'questions', tip: 'Touch forehead then wiggle Y hand.' },
  { word: 'how', category: 'questions', tip: 'Curved hands twist together.' },
  { word: 'which', category: 'questions', tip: 'Both index fingers alternate.' },
  { word: 'can', category: 'verbs', tip: 'Fists rotate down together.' },
  { word: 'cannot', category: 'verbs', tip: 'Middle fingers snap apart.' },
  { word: 'go', category: 'verbs', tip: 'Pointing hands move forward.' },
  { word: 'come', category: 'verbs', tip: 'Hand beckons toward body.' },
  { word: 'give', category: 'verbs', tip: 'Flat hand moves outward.' },
  { word: 'take', category: 'verbs', tip: 'Claw hand pulls inward.' },
  { word: 'make', category: 'verbs', tip: 'Fists twist together.' },
  { word: 'read', category: 'verbs', tip: 'V fingers track along palm.' },
  { word: 'write', category: 'verbs', tip: 'Index writes on opposite palm.' },
  { word: 'learn', category: 'verbs', tip: 'Palm-to-forehead scoop movement.' },
  { word: 'understand', category: 'verbs', tip: 'Index pops up near temple.' },
  { word: 'remember', category: 'verbs', tip: 'Touch forehead then fist closes.' },
  { word: 'forget', category: 'verbs', tip: 'Swipe forehead with thumb.' },
  { word: 'know', category: 'verbs', tip: 'Fingers tap forehead.' },
  { word: 'like', category: 'feelings', tip: 'Middle finger and thumb pull from chest.' },
  { word: 'love', category: 'feelings', tip: 'Cross hands over chest.' },
  { word: 'happy', category: 'feelings', tip: 'Flat hands brush upward on chest.' },
  { word: 'sad', category: 'feelings', tip: 'Hands slide down face.' },
  { word: 'angry', category: 'feelings', tip: 'Claw hands rise with tense face.' },
  { word: 'excited', category: 'feelings', tip: 'Hands shake near chest.' },
  { word: 'tired', category: 'feelings', tip: 'Bent hands drop down.' },
  { word: 'hot', category: 'feelings', tip: 'Pinched hand opens away from mouth.' },
  { word: 'cold', category: 'feelings', tip: 'Fists shake near body.' },
  { word: 'good', category: 'descriptions', tip: 'Flat hand from chin to palm.' },
  { word: 'bad', category: 'descriptions', tip: 'Flat hand from chin turns down.' },
  { word: 'big', category: 'descriptions', tip: 'Hands spread apart.' },
  { word: 'small', category: 'descriptions', tip: 'Pinch gap between fingers.' },
  { word: 'fast', category: 'descriptions', tip: 'Hands flick past each other.' },
  { word: 'slow', category: 'descriptions', tip: 'One hand moves gradually over other.' },
  { word: 'easy', category: 'descriptions', tip: 'Hands brush off palms.' },
  { word: 'difficult', category: 'descriptions', tip: 'Bent index fingers twist.' },
  { word: 'clean', category: 'descriptions', tip: 'Brush one palm over other.' },
  { word: 'dirty', category: 'descriptions', tip: 'Back of hand brushes chin.' },
  { word: 'today', category: 'time', tip: 'Y hands move down together.' },
  { word: 'tomorrow', category: 'time', tip: 'Thumb from cheek moves forward.' },
  { word: 'yesterday', category: 'time', tip: 'Thumb from cheek moves backward.' },
  { word: 'now', category: 'time', tip: 'Bent index fingers point down.' },
  { word: 'later', category: 'time', tip: 'L hand flips near shoulder.' },
  { word: 'morning', category: 'time', tip: 'Arm rises like sunrise.' },
  { word: 'afternoon', category: 'time', tip: 'Arm moves to horizon level.' },
  { word: 'night', category: 'time', tip: 'Arm closes over other arm.' },
  { word: 'week', category: 'time', tip: 'Index slides across palm.' },
  { word: 'month', category: 'time', tip: 'Index travels down index of other hand.' },
  { word: 'red', category: 'colors', tip: 'Index brushes lip.' },
  { word: 'blue', category: 'colors', tip: 'B hand shakes sideward.' },
  { word: 'green', category: 'colors', tip: 'G hand shakes lightly.' },
  { word: 'yellow', category: 'colors', tip: 'Y hand wiggles.' },
  { word: 'black', category: 'colors', tip: 'Index draws across forehead.' },
  { word: 'white', category: 'colors', tip: 'Open hand pulls from chest.' },
  { word: 'orange', category: 'colors', tip: 'S hand opens and closes by chin.' },
  { word: 'pink', category: 'colors', tip: 'Middle finger brushes chin.' },
  { word: 'brown', category: 'colors', tip: 'B hand drags down cheek.' },
  { word: 'purple', category: 'colors', tip: 'P hand shakes.' },
];

/* ============================================================
   DOM References
   ============================================================ */
const $ = id => document.getElementById(id);
const video         = $('video');
const canvas        = $('canvas');
const ctx           = canvas.getContext('2d');
const placeholder   = $('camera-placeholder');
const cameraFrame   = $('camera-frame');
const cameraHud     = $('camera-hud');
const cameraCtrl    = $('camera-controls');
const hudFps        = $('hud-fps');
const hudHandInfo   = $('hud-hand-info');
const detectionBadge= $('detection-badge');
const detectionLetter = $('detection-letter');
const bigLetter     = $('big-letter');
const confidenceBar = $('confidence-bar');
const confidenceLabel = $('confidence-label');
const letterDesc    = $('letter-description');
const wordDisplay   = $('word-display');
const debounceBar   = $('debounce-bar');
const sentenceDisplay = $('sentence-display');
const statLetters   = $('stat-letters');
const statWords     = $('stat-words');
const statAccuracy  = $('stat-accuracy');
const statTime      = $('stat-time');
const toast         = $('toast');
const bslSearch     = $('bsl-search');
const bslGuideGrid  = $('bsl-guide-grid');
const debugVelocity = $('debug-velocity');
const debugVote     = $('debug-vote');
const debugHold     = $('debug-hold');
const debugCooldown = $('debug-cooldown');

/* ============================================================
   State
   ============================================================ */
const state = {
  cameraActive:     false,
  stream:           null,
  tracker:          null,     // MediaPipe Holistic instance
  mpCamera:         null,     // MediaPipe Camera utils instance
  animFrame:        null,
  showLandmarks:    true,
  facingMode:       'user',

  currentWord:      null,
  currentConf:      0,
  stableWord:       null,
  stableStart:      0,
  lastCommitted:    '',
  lastCommitTime:   0,
  predictionWindow: [],
  prevLandmarks:    null,
  prevFrameTs:      0,
  handPresent:      false,

  word:             '',
  sentence:         '',
  letterCount:      0,
  wordCount:        0,
  confSamples:      [],
  sessionStart:     null,
  sessionTimer:     null,

  fps:              0,
  lastFrameTime:    0,
  frameCount:       0,
  fpsInterval:      null,
};

/* ============================================================
   BSL Word Classifier (Limited common words)
   ============================================================ */

/**
 * Normalize landmarks: shift by wrist, scale by hand size.
 */
function normalizeLandmarks(landmarks) {
  const wrist = landmarks[0];
  const midBase = landmarks[9]; // Middle base
  const scale = Math.sqrt(
    Math.pow(midBase.x - wrist.x, 2) +
    Math.pow(midBase.y - wrist.y, 2)
  ) || 1;

  return landmarks.map(lm => ({
    x: (lm.x - wrist.x) / scale,
    y: (lm.y - wrist.y) / scale,
    z: (lm.z - wrist.z) / scale,
  }));
}

/**
 * Returns how "extended" each finger is: [thumb, index, middle, ring, pinky]
 * 1 = extended, 0 = curled, 0.5 = ambiguous.
 *
 * Orientation-invariant: checks the angle of the finger at the PIP joint by
 * comparing the MCP→PIP direction with the PIP→TIP direction. This works when
 * the hand points up, down, sideways, or toward the camera.
 */
function getFingerStates(lm) {
  const fingerTips = [4, 8, 12, 16, 20];
  const fingerPips = [3, 6, 10, 14, 18];
  const fingerMcps = [2, 5,  9, 13, 17];

  // Thumb: angle at IP (index 3). Vectors MCP(2)→IP(3) and IP(3)→TIP(4).
  // If the thumb is folded across the palm, those vectors point oppositely (dot < 0).
  const t_v1 = { x: lm[3].x - lm[2].x, y: lm[3].y - lm[2].y };
  const t_v2 = { x: lm[4].x - lm[3].x, y: lm[4].y - lm[3].y };
  const t_dot = t_v1.x * t_v2.x + t_v1.y * t_v2.y;
  const t_n1 = Math.hypot(t_v1.x, t_v1.y) || 1;
  const t_n2 = Math.hypot(t_v2.x, t_v2.y) || 1;
  const t_cos = t_dot / (t_n1 * t_n2);
  // cos > 0.3 → roughly aligned → extended; cos < -0.1 → folded back → curled.
  const thumbExtended = t_cos > 0.3 ? 1 : (t_cos < -0.1 ? 0 : 0.5);

  const extended = [thumbExtended];
  for (let i = 1; i < 5; i++) {
    const tip = lm[fingerTips[i]];
    const pip = lm[fingerPips[i]];
    const mcp = lm[fingerMcps[i]];
    const v1 = { x: pip.x - mcp.x, y: pip.y - mcp.y };
    const v2 = { x: tip.x - pip.x, y: tip.y - pip.y };
    const dot = v1.x * v2.x + v1.y * v2.y;
    const n1 = Math.hypot(v1.x, v1.y) || 1;
    const n2 = Math.hypot(v2.x, v2.y) || 1;
    const cos = dot / (n1 * n2);
    // cos > 0.48 → finger continues straight → extended
    // cos < -0.05 → finger curled back → curled
    // 0.48 is a compromise: tight enough that a soft fist (~60° bend) reads
    // as curled, loose enough that a fast-moving open palm (during a wave)
    // still reads as extended even with noisy MediaPipe frames.
    const ext = cos > 0.48 ? 1 : (cos < -0.05 ? 0 : 0);
    extended.push(ext);
  }
  return extended;
}

/** Euclidean distance between two 2D landmark points */
function dist2d(a, b) {
  return Math.sqrt(Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2));
}

/** Are two landmarks close together? */
function close(a, b, thresh = 0.25) {
  return dist2d(a, b) < thresh;
}

function mirrorLandmarks(landmarks) {
  if (!Array.isArray(landmarks)) return null;
  return landmarks.map(lm => ({ x: 1 - lm.x, y: lm.y, z: lm.z, visibility: lm.visibility }));
}

function isValidHandLandmarks(landmarks) {
  return Array.isArray(landmarks) && landmarks.length >= 21;
}

/** Prefer the hand closest to the horizontal centre of the frame (usually the signing hand in selfie view). */
function pickPrimaryMirroredHand(rawLeft, rawRight) {
  if (!rawLeft) return rawRight;
  if (!rawRight) return rawLeft;
  const dl = Math.abs(rawLeft[0].x - 0.5);
  const dr = Math.abs(rawRight[0].x - 0.5);
  return dl <= dr ? rawLeft : rawRight;
}

function bothHandsMostlyExtended(leftLm, rightLm) {
  const [tl, il, ml, rl, pl] = getFingerStates(leftLm);
  const [tr, ir, mr, rr, pr] = getFingerStates(rightLm);
  const leftOk = il >= 0.5 && ml >= 0.5 && rl >= 0.5 && pl >= 0.5 && tl >= 0.5;
  const rightOk = ir >= 0.5 && mr >= 0.5 && rr >= 0.5 && pr >= 0.5 && tr >= 0.5;
  return leftOk && rightOk;
}

/**
 * Unified BSL classifier — receives BOTH hands, face, and pose simultaneously.
 * This is the core fix: two-hand signs are now classified as a whole gesture,
 * not as two separate single-hand classifications.
 *
 * @param {Array|null} rawLeft  – mirrored left-hand landmarks (21 pts)
 * @param {Array|null} rawRight – mirrored right-hand landmarks (21 pts)
 * @param {Array|null} faceLm  – mirrored face landmarks
 * @param {Array|null} poseLm  – mirrored pose landmarks
 */
function classifyBSLWordBothHands(rawLeft, rawRight, faceLm, poseLm) {
  const primaryRaw = pickPrimaryMirroredHand(rawLeft, rawRight);

  if (!primaryRaw) {
    return { word: null, confidence: 0, margin: 0, motionLandmarks: null };
  }

  const lm   = normalizeLandmarks(primaryRaw);
  const [thumb, index, middle, ring, pinky] = getFingerStates(lm);

  const fingers           = [index, middle, ring, pinky];
  const allFingersCurled   = fingers.every(f => f === 0);
  const allFingersExtended = fingers.every(f => f === 1);

  // Raw coords for face/pose proximity checks
  const thumbTipRaw  = primaryRaw[4];
  const indexTipRaw  = primaryRaw[8];
  const middleTipRaw = primaryRaw[12];
  const ringTipRaw   = primaryRaw[16];
  const pinkyTipRaw  = primaryRaw[20];
  const wristRaw     = primaryRaw[0];

  // Use the palm centre (middle-finger MCP) for horizontal centring rather than the
  // thumb tip — for "hello" the thumb is off to the side even though the hand is centred.
  const palmCenter = primaryRaw[9];
  const nearFaceCenter   = palmCenter.x > 0.22 && palmCenter.x < 0.78;
  const isUpperFrame     = wristRaw.y < 0.62;
  const isVeryUpperFrame = wristRaw.y < 0.42;

  const chinLm        = faceLm?.[152] || null;
  const foreheadLm    = faceLm?.[10]  || null;
  const nosePoseLm    = poseLm?.[0]   || null;
  const leftShouLm    = poseLm?.[11]  || null;
  const rightShouLm   = poseLm?.[12]  || null;
  const chestCenter   = (leftShouLm && rightShouLm)
    ? { x: (leftShouLm.x + rightShouLm.x) / 2, y: (leftShouLm.y + rightShouLm.y) / 2 + 0.07 }
    : null;

  const nearChinByFace      = chinLm    ? dist2d(thumbTipRaw, chinLm)    < 0.16 : false;
  const veryNearChinByFace  = chinLm    ? dist2d(thumbTipRaw, chinLm)    < 0.12 : false;
  const nearForeheadByFace  = foreheadLm? dist2d(thumbTipRaw, foreheadLm)< 0.18 : false;
  const nearHeadByPose      = nosePoseLm? dist2d(thumbTipRaw, nosePoseLm)< 0.26 : false;
  const nearChest           = chestCenter? dist2d(wristRaw, chestCenter)  < 0.26 : false;
  const veryNearChest       = chestCenter? dist2d(wristRaw, chestCenter)  < 0.18 : false;

  const avgFingerTip = {
    x: (indexTipRaw.x + middleTipRaw.x + ringTipRaw.x + pinkyTipRaw.x) / 4,
    y: (indexTipRaw.y + middleTipRaw.y + ringTipRaw.y + pinkyTipRaw.y) / 4,
  };
  const nearChinByFingers     = chinLm     ? dist2d(avgFingerTip, chinLm)     < 0.20 : false;
  const nearForeheadByFingers = foreheadLm ? dist2d(avgFingerTip, foreheadLm) < 0.22 : false;
  const nearMouthByFingers    = chinLm     ? dist2d(avgFingerTip, chinLm)     < 0.16 : false;
  const handBelowForehead = foreheadLm ? wristRaw.y > (foreheadLm.y + 0.06) : true;
  const handAboveChest    = chestCenter ? wristRaw.y < (chestCenter.y + 0.05) : true;
  const handNearFaceZone  = nearChinByFace || nearChinByFingers || nearForeheadByFace || nearHeadByPose;

  const thumbUp     = lm[4].y < lm[3].y - 0.04;
  const thumbOutward = Math.abs(lm[4].x - lm[3].x) > 0.25;

  // ── Two-hand helper data ──────────────────────────────────────
  const bothHands = rawLeft && rawRight;
  let twoHandDist = Infinity;
  let leftLm = null;
  let rightLm = null;
  let bothExtended = false;
  let leftCurled = false;
  let rightCurled = false;

  if (bothHands) {
    twoHandDist = dist2d(rawLeft[0], rawRight[0]);
    leftLm = normalizeLandmarks(rawLeft);
    rightLm = normalizeLandmarks(rawRight);
    bothExtended = bothHandsMostlyExtended(leftLm, rightLm);
    leftCurled = getFingerStates(leftLm).slice(1).every(f => f === 0);
    rightCurled = getFingerStates(rightLm).slice(1).every(f => f === 0);
  }

  // ── Score each word ───────────────────────────────────────────
  const scores = {};

  // Face-anchored vertical bands. When face landmarks aren't available we
  // fall back to fixed image-coordinate bands that work for a typical
  // upper-body webcam framing.
  let foreheadAnchor = 0.20;
  let chinAnchor     = 0.40;
  let chestAnchor    = 0.60;
  if (foreheadLm && chinLm) {
    foreheadAnchor = foreheadLm.y;
    chinAnchor     = chinLm.y;
    const faceH    = Math.max(0.10, chinLm.y - foreheadLm.y);
    chestAnchor    = chinLm.y + faceH * 1.1;
  }
  const faceH = Math.max(0.10, chinAnchor - foreheadAnchor);

  // The "contact point" of a sign matters more than the wrist:
  // hello/father → fingertips/thumb at FOREHEAD; thank-you/mother → at CHIN.
  const fyTip = avgFingerTip.y;
  const tyTip = thumbTipRaw.y;
  const wy    = wristRaw.y;

  // Wrist can hang well below the contact point of a sign (e.g. wrist near jaw
  // when fingertips touch the forehead), so the bands here are generous.
  const wristAtForehead = wy < chinAnchor + faceH * 0.2;
  const wristAtChin     = wy >= foreheadAnchor + faceH * 0.4 && wy < chinAnchor + faceH * 0.8;
  const wristAtChest    = wy >= chinAnchor + faceH * 0.5;
  const handOffCentre   = !nearFaceCenter && wy >= foreheadAnchor + faceH * 0.3 && wy < chinAnchor + faceH * 1.2;

  // Specific contact tests (used to pick between similar signs).
  const fingersAtForehead = (fyTip < foreheadAnchor + faceH * 0.4) && nearFaceCenter;
  const thumbAtForehead   = nearForeheadByFace || ((tyTip < foreheadAnchor + faceH * 0.3) && nearFaceCenter);
  const fingersAtChin     = ((fyTip >= chinAnchor - faceH * 0.3) && (fyTip < chinAnchor + faceH * 0.5)) && nearFaceCenter;
  const thumbAtChin       = veryNearChinByFace || ((tyTip >= chinAnchor - faceH * 0.25) && (tyTip < chinAnchor + faceH * 0.35) && nearFaceCenter);

  // Handshape signatures. With the new orientation-invariant finger test we
  // allow a small amount of slack (3-of-4 fingers extended is still "open palm")
  // so a single mis-detected finger doesn't tank the whole sign.
  const extendedCount = fingers.filter(f => f === 1).length;
  const curledCount   = fingers.filter(f => f === 0).length;
  const openHand      = extendedCount >= 3;
  const closedHand    = curledCount >= 3;

  const openPalm   = openHand && thumb >= 0.5;
  // Strict thumb states: a thumb registered as ambiguous (0.5) should NOT
  // satisfy *both* fist and thumbsUp at once — that was causing "sorry"
  // (fist) frames to also score "yes" (thumbs-up).
  const fist       = closedHand && thumb < 0.5;    // thumb folded over fingers
  const thumbsUp   = closedHand && thumb > 0.5;    // closed fist, thumb extended
  const cHand      = thumb >= 0.5 && index === 1 && middle === 1 && ring === 0 && pinky === 0;
  // Tightened from 0.35 → 0.26: a real "eat" pinch is a tight O-shape.
  const pinch      = close(lm[4], lm[8], 0.26);
  const twoFinger  = index === 1 && middle === 1 && ring === 0 && pinky === 0;

  // Face-region contacts (rely on face landmarks when present, otherwise fall
  // back to face-anchored y-bands defined above).
  const inForeheadArea = (nearForeheadByFace || nearForeheadByFingers || nearHeadByPose)
                         || (wristAtForehead && nearFaceCenter);
  const inChinArea     = (veryNearChinByFace || nearChinByFingers)
                         || (wristAtChin && nearFaceCenter);
  const handAtMouth    = veryNearChinByFace || nearMouthByFingers
                         || (wristAtChin && nearFaceCenter);

  // ── Relative-dominance contact tests ─────────────────────────
  // For hello/father (forehead) and thank-you/mother (chin) the signs are
  // *identical* except for whether fingertips or thumb-pad contacts the
  // face. Pick whichever contact is *closer* to the anchor.
  const distThumbToChin       = chinLm     ? dist2d(thumbTipRaw,  chinLm)     : Infinity;
  const distFingersToChin     = chinLm     ? dist2d(avgFingerTip, chinLm)     : Infinity;
  const distThumbToForehead   = foreheadLm ? dist2d(thumbTipRaw,  foreheadLm) : Infinity;
  const distFingersToForehead = foreheadLm ? dist2d(avgFingerTip, foreheadLm) : Infinity;

  // ── Wrist-Y based zones (PRIMARY zone discriminator) ─────────
  // Critical: zone is decided by WRIST Y, not fingertip proximity. An open
  // palm on the chest with fingers pointing up has its fingertips very
  // close to the chin landmark — using fingertip distance for zoning
  // wrongly labels that "at chin" and blocks please/sorry.
  const wristZoneIsForehead = wy < foreheadAnchor + faceH * 0.5;
  const wristZoneIsChin     = wy >= foreheadAnchor + faceH * 0.5 && wy < chinAnchor + faceH * 0.45;
  const wristZoneIsChest    = wy >= chinAnchor + faceH * 0.45;

  // Pose-based chest indicator (most reliable when pose landmarks present).
  const handOnChest = nearChest
    || (chestCenter && wy >= chestCenter.y - 0.06)
    || (!chestCenter && wristZoneIsChest);

  // Face-zone gates: require the WRIST to actually be at that zone *and*
  // some part of the hand near the corresponding face landmark.
  const openPalmAtChinZone = wristZoneIsChin
    && nearFaceCenter
    && (nearChinByFingers || veryNearChinByFace);
  const openPalmAtForeheadZone = wristZoneIsForehead
    && (nearForeheadByFingers || nearForeheadByFace || nearHeadByPose || nearFaceCenter);

  // Strict thumb-dominance margin: father/mother only fire when the thumb
  // is *clearly* closer to the anchor than the fingertips. Otherwise the
  // default (hello/thank-you) wins. Without this margin, a saluting wave
  // could land in a tie and both signs would collapse to 0.1.
  const THUMB_DOMINANCE_MARGIN = 0.03;
  const thumbClearlyAtForehead = distThumbToForehead < distFingersToForehead - THUMB_DOMINANCE_MARGIN
                                  && nearForeheadByFace;
  const thumbClearlyAtChin     = distThumbToChin     < distFingersToChin     - THUMB_DOMINANCE_MARGIN
                                  && veryNearChinByFace;

  // HELLO — open palm at brow. Default winner unless thumb clearly dominates.
  scores.hello = (
    openPalm && openPalmAtForeheadZone && !thumbClearlyAtForehead
  ) ? 0.88 : 0.1;

  // FATHER — open-B hand, thumb pad pressed against the forehead.
  scores.father = (
    openPalm && openPalmAtForeheadZone && thumbClearlyAtForehead
  ) ? 0.92 : 0.1;

  // THANK YOU — open palm fingertips at chin. Default winner at the chin zone.
  scores['thank you'] = (
    openPalm && openPalmAtChinZone && !thumbClearlyAtChin
  ) ? 0.88 : 0.1;

  // MOTHER — open-B hand, thumb pad at chin.
  scores.mother = (
    openPalm && openPalmAtChinZone && thumbClearlyAtChin
  ) ? 0.92 : 0.1;

  // GOODBYE — open palm beside the head/shoulder (the "wave" gesture).
  // Relaxed from strict handOffCentre: it's enough that the palm is not
  // pressed against the face zone and the hand is in upper-body range.
  const handBesideFace = palmCenter.x < 0.32 || palmCenter.x > 0.68;
  const inWaveBand     = wy >= foreheadAnchor && wy < chinAnchor + faceH * 1.3;
  scores.goodbye = (
    openPalm
    && inWaveBand
    && (handBesideFace || handOffCentre)
    && !inChinArea
    && !nearForeheadByFingers   // avoid stealing hello frames
    && !nearForeheadByFace
  ) ? 0.84 : 0.1;

  // YES — closed fist with thumb extended along the side (nodding motion).
  // Bumped 0.80 → 0.86 so the margin over any neighbour (e.g. help) stays
  // well above MIN_SCORE_MARGIN.
  scores.yes = thumbsUp ? 0.86 : 0.1;

  // NO — index + middle extended, tip pinches thumb.
  scores.no = (twoFinger && close(lm[8], lm[4], 0.32)) ? 0.84 : 0.1;

  // PLEASE — open palm on the chest. Zoning is wrist-based so an open palm
  // at chest (fingers pointing up) no longer gets falsely flagged as "near
  // chin" via fingertip proximity.
  scores.please = (
    openPalm && handOnChest && !pinch
  ) ? 0.88 : 0.1;

  // SORRY — closed fist on the chest.
  scores.sorry = (
    fist && handOnChest
  ) ? 0.90 : 0.1;

  // EAT — pinch shape near mouth. Wrist must be at chin level (not chest).
  scores.eat = (
    pinch && wristZoneIsChin && nearFaceCenter && !openPalm
  ) ? 0.86 : 0.1;

  // DRINK — C-hand near mouth.
  scores.drink = (
    cHand && wristZoneIsChin && nearFaceCenter
  ) ? 0.82 : 0.1;

  // ── Two-hand signs (only score high if both hands are actually visible) ──
  // HELP: one hand (thumb-up fist) rests on a flat palm and lifts.
  // The previous single-hand fallback (0.68) collided with YES (thumbs-up),
  // killing the score margin and rejecting both. HELP now strictly requires
  // both hands.
  scores.help = bothHands
    && twoHandDist < 0.45
    && (leftCurled || rightCurled)
    && thumbsUp
    ? 0.86 : 0.1;

  // WANT: both hands open, palms up, pulled toward body
  scores.want = bothHands && twoHandDist < 0.6
    && bothExtended
    && rawLeft[0].y > 0.4 && rawRight[0].y > 0.4
    ? 0.80 : 0.1;

  // FRIEND: both index fingers hook together (wrists close, both curled)
  scores.friend = bothHands && twoHandDist < 0.3 && leftCurled && rightCurled
    ? 0.78 : 0.1;

  // WORK: two fists tapping (wrists close, both curled)
  scores.work = bothHands && twoHandDist < 0.35 && leftCurled && rightCurled
    && rawLeft[0].y > 0.38   // hands at mid-body level
    ? 0.76 : 0.1;

  // SCHOOL / CLAP — both palms come together side-by-side (similar wrist height)
  scores.school = bothHands && twoHandDist < 0.25 && bothExtended
    && Math.abs(rawLeft[0].y - rawRight[0].y) < 0.07
    ? 0.78 : 0.1;

  // STOP — dominant flat hand chops onto the OTHER flat palm.
  //  Required: both hands open, one above the other (clear vertical offset),
  //  wrists roughly aligned horizontally so it reads as a chop.
  scores.stop = bothHands
    && twoHandDist < 0.32
    && bothExtended
    && Math.abs(rawLeft[0].y - rawRight[0].y) > 0.08
    && Math.abs(rawLeft[0].x - rawRight[0].x) < 0.18
    ? 0.84 : 0.1;

  // FAMILY: F hands circle together (both extended, close to each other)
  scores.family = bothHands && twoHandDist < 0.40 && bothExtended
    && rawLeft[0].y < 0.65
    ? 0.76 : 0.1;

  // ── Rank and return ───────────────────────────────────────────
  const ranked = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const [bestWord, bestScore] = ranked[0] || [null, 0];
  const [, secondScore]       = ranked[1] || [null, 0];

  // Build a human-readable handshape tag for the debug overlay.
  let handshapeTag = 'unknown';
  if (openPalm)        handshapeTag = 'open palm';
  else if (fist)       handshapeTag = 'fist';
  else if (thumbsUp)   handshapeTag = 'thumbs-up fist';
  else if (cHand)      handshapeTag = 'C-hand';
  else if (twoFinger)  handshapeTag = '2-finger';
  else if (pinch)      handshapeTag = 'pinch';

  let zoneTag = 'mid';
  if      (wristAtForehead) zoneTag = 'forehead';
  else if (wristAtChin)     zoneTag = 'chin';
  else if (wristAtChest)    zoneTag = 'chest';
  else if (handOffCentre)   zoneTag = 'side';

  return {
    word: bestWord,
    confidence: bestScore,
    margin: bestScore - secondScore,
    motionLandmarks: primaryRaw,
    debug: {
      handshapeTag,
      zoneTag,
      foreheadAnchor,
      chinAnchor,
      chestAnchor,
      extendedCount,
    },
  };
}

/* ============================================================
   Temporal Smoothing + Gesture Stability Pipeline
   ============================================================ */
function resetPredictionWindow() {
  state.predictionWindow = [];
}

function pushPrediction(word, confidence) {
  state.predictionWindow.push({ word, confidence, ts: Date.now() });
  if (state.predictionWindow.length > CONFIG.PREDICTION_WINDOW_SIZE) {
    state.predictionWindow.shift();
  }
}

function getMajorityVote() {
  if (state.predictionWindow.length === 0) {
    return { word: null, ratio: 0, avgConfidence: 0 };
  }

  const counts = new Map();
  const confSums = new Map();
  for (const row of state.predictionWindow) {
    counts.set(row.word, (counts.get(row.word) || 0) + 1);
    confSums.set(row.word, (confSums.get(row.word) || 0) + row.confidence);
  }

  let bestWord = null;
  let bestCount = 0;
  for (const [word, count] of counts.entries()) {
    if (count > bestCount) {
      bestWord = word;
      bestCount = count;
    }
  }

  if (!bestWord) return { word: null, ratio: 0, avgConfidence: 0 };
  const ratio = bestCount / state.predictionWindow.length;
  const avgConfidence = (confSums.get(bestWord) || 0) / bestCount;
  return { word: bestWord, ratio, avgConfidence };
}

function getWordConfidenceThreshold(word) {
  if (!word) return CONFIG.MIN_WORD_CONFIDENCE;
  return WORD_MIN_CONFIDENCE[word] || CONFIG.MIN_WORD_CONFIDENCE;
}

function getAverageVelocity(landmarks) {
  const now = Date.now();
  if (!state.prevLandmarks || !state.prevFrameTs) {
    state.prevLandmarks = landmarks.map(l => ({ ...l }));
    state.prevFrameTs = now;
    return 0;
  }

  const dt = Math.max((now - state.prevFrameTs) / 1000, 0.001);
  let totalVel = 0;
  const sampleIdx = [0, 4, 8, 12, 16, 20];
  for (const idx of sampleIdx) {
    const prev = state.prevLandmarks[idx];
    const curr = landmarks[idx];
    if (!prev || !curr) continue;
    const d = dist2d(prev, curr);
    totalVel += d / dt;
  }

  state.prevLandmarks = landmarks.map(l => ({ ...l }));
  state.prevFrameTs = now;
  return totalVel / sampleIdx.length;
}

function resetStabilityTracking() {
  state.stableWord = null;
  state.stableStart = 0;
  debounceBar.style.width = '0%';
}

function updateDebugStrip({ velocity = 0, voteRatio = 0, holdProgress = 0, inCooldown = false } = {}) {
  if (debugVelocity) debugVelocity.textContent = velocity.toFixed(2);
  if (debugVote) debugVote.textContent = `${Math.round(voteRatio * 100)}%`;
  if (debugHold) debugHold.textContent = `${Math.round(holdProgress * 100)}%`;
  if (debugCooldown) debugCooldown.textContent = inCooldown ? 'cooldown' : 'ready';
}

/* ============================================================
   Canvas Rendering
   ============================================================ */

/** Get bounding box from landmarks in canvas space */
function getBoundingBox(landmarks, cw, ch) {
  const xs = landmarks.map(l => l.x * cw);
  const ys = landmarks.map(l => l.y * ch);
  const pad = 24;
  return {
    x:  Math.min(...xs) - pad,
    y:  Math.min(...ys) - pad,
    x2: Math.max(...xs) + pad,
    y2: Math.max(...ys) + pad,
  };
}

function getBoundingBoxUnion(lmA, lmB, cw, ch) {
  if (lmA && lmB) {
    const bbA = getBoundingBox(lmA, cw, ch);
    const bbB = getBoundingBox(lmB, cw, ch);
    return {
      x: Math.min(bbA.x, bbB.x),
      y: Math.min(bbA.y, bbB.y),
      x2: Math.max(bbA.x2, bbB.x2),
      y2: Math.max(bbA.y2, bbB.y2),
    };
  }
  return getBoundingBox(lmA || lmB, cw, ch);
}

/** Draw corner-style bounding box */
function drawBoundingBox(bb, label) {
  const { x, y, x2, y2 } = bb;
  const w = x2 - x;
  const h = y2 - y;
  const cl = CONFIG.BBOX_CORNER_LEN;

  ctx.strokeStyle = CONFIG.BBOX_COLOR;
  ctx.lineWidth   = CONFIG.BBOX_LINE_WIDTH;
  ctx.lineCap     = 'round';

  ctx.beginPath();
  // TL
  ctx.moveTo(x + cl, y); ctx.lineTo(x, y); ctx.lineTo(x, y + cl);
  // TR
  ctx.moveTo(x2 - cl, y); ctx.lineTo(x2, y); ctx.lineTo(x2, y + cl);
  // BL
  ctx.moveTo(x, y2 - cl); ctx.lineTo(x, y2); ctx.lineTo(x + cl, y2);
  // BR
  ctx.moveTo(x2 - cl, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - cl);
  ctx.stroke();

  // Label background
  if (label) {
    const fontSize = 15;
    ctx.font = `700 ${fontSize}px Inter, sans-serif`;
    const tw = ctx.measureText(label).width;
    const lx = x;
    const ly = y - 8;
    ctx.fillStyle = 'rgba(6,182,212,0.85)';
    ctx.beginPath();
    ctx.roundRect(lx, ly - fontSize - 4, tw + 16, fontSize + 10, 6);
    ctx.fill();
    ctx.fillStyle = 'white';
    ctx.fillText(label, lx + 8, ly);
  }
}

/** Draw landmarks and connectors */
function drawLandmarks(landmarks, cw, ch) {
  if (!state.showLandmarks) return;

  // MediaPipe hand connections
  const CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],           // Thumb
    [0,5],[5,6],[6,7],[7,8],           // Index
    [0,9],[9,10],[10,11],[11,12],      // Middle
    [0,13],[13,14],[14,15],[15,16],    // Ring
    [0,17],[17,18],[18,19],[19,20],    // Pinky
    [5,9],[9,13],[13,17],              // Palm
  ];

  // Draw connectors
  ctx.strokeStyle = CONFIG.CONNECTOR_COLOR;
  ctx.lineWidth = 2;
  ctx.lineCap = 'round';
  for (const [a, b] of CONNECTIONS) {
    ctx.beginPath();
    ctx.moveTo(landmarks[a].x * cw, landmarks[a].y * ch);
    ctx.lineTo(landmarks[b].x * cw, landmarks[b].y * ch);
    ctx.stroke();
  }

  // Draw landmark dots
  for (let i = 0; i < landmarks.length; i++) {
    const lm = landmarks[i];
    const px = lm.x * cw;
    const py = lm.y * ch;
    const isTip = [4, 8, 12, 16, 20].includes(i);

    ctx.beginPath();
    ctx.arc(px, py, isTip ? 6 : 4, 0, Math.PI * 2);
    ctx.fillStyle = isTip
      ? `rgba(6,182,212,0.95)`
      : `rgba(124,58,237,0.9)`;
    ctx.fill();

    // Outer ring for tips
    if (isTip) {
      ctx.beginPath();
      ctx.arc(px, py, 9, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(6,182,212,0.4)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }
}

/** Draw faint horizontal zone bands so the user can see what the detector is using. */
function drawZoneBands(debug, cw, ch) {
  const { foreheadAnchor, chinAnchor, chestAnchor } = debug;
  const faceH = Math.max(0.10, chinAnchor - foreheadAnchor);

  const bands = [
    { y0: 0,                                 y1: chinAnchor + faceH * 0.2,  color: 'rgba(167,139,250,0.07)', label: 'forehead' },
    { y0: foreheadAnchor + faceH * 0.4,      y1: chinAnchor + faceH * 0.8,  color: 'rgba(244,114,182,0.07)', label: 'chin' },
    { y0: chinAnchor + faceH * 0.5,          y1: 1,                          color: 'rgba(16,185,129,0.07)',  label: 'chest' },
  ];

  ctx.save();
  for (const b of bands) {
    ctx.fillStyle = b.color;
    const y = b.y0 * ch;
    const h = (b.y1 - b.y0) * ch;
    ctx.fillRect(0, y, cw, h);
    ctx.fillStyle = b.color.replace(/0\.07/, '0.55');
    ctx.font = '600 11px Inter, sans-serif';
    ctx.fillText(b.label, 8, y + 14);
  }
  ctx.restore();
}

/** Render a single frame onto the canvas */
function renderFrame(results) {
  const cw = canvas.width;
  const ch = canvas.height;

  // Clear
  ctx.clearRect(0, 0, cw, ch);

  // Draw mirrored video frame
  ctx.save();
  ctx.scale(-1, 1);
  ctx.drawImage(results.image, -cw, 0, cw, ch);
  ctx.restore();

  const mLeft = isValidHandLandmarks(results.leftHandLandmarks)
    ? mirrorLandmarks(results.leftHandLandmarks)
    : null;
  const mRight = isValidHandLandmarks(results.rightHandLandmarks)
    ? mirrorLandmarks(results.rightHandLandmarks)
    : null;
  const mirroredFace = mirrorLandmarks(results.faceLandmarks);
  const mirroredPose = mirrorLandmarks(results.poseLandmarks);

  if (mLeft || mRight) {
    if (mLeft) drawLandmarks(mLeft, cw, ch);
    if (mRight) drawLandmarks(mRight, cw, ch);

    const result = classifyBSLWordBothHands(mLeft, mRight, mirroredFace, mirroredPose);
    const { word, confidence, margin, motionLandmarks, debug } = result;

    if (debug && state.showLandmarks) drawZoneBands(debug, cw, ch);

    const bb = getBoundingBoxUnion(mLeft, mRight, cw, ch);
    const labelStr = confidence >= CONFIG.MIN_WORD_CONFIDENCE
      ? `${word}  ${Math.round(confidence * 100)}%`
      : `${debug?.handshapeTag || '?'} @ ${debug?.zoneTag || '?'}`;
    drawBoundingBox(bb, labelStr);

    state.handPresent = true;
    if (motionLandmarks) {
      processWordDetection(word, confidence, margin, motionLandmarks);
    }

    const count = (mLeft ? 1 : 0) + (mRight ? 1 : 0);
    const faceLabel = mirroredFace ? 'face' : 'no-face';
    const poseLabel = mirroredPose ? 'pose' : 'no-pose';
    hudHandInfo.textContent = `${count} hand · ${debug?.handshapeTag || '?'} · zone: ${debug?.zoneTag || '?'} · ${faceLabel} · ${poseLabel}`;
  } else {
    const faceLabel = mirroredFace ? 'face' : 'no-face';
    const poseLabel = mirroredPose ? 'pose' : 'no-pose';
    hudHandInfo.textContent = `Scanning for hand... (${faceLabel}, ${poseLabel})`;
    handleNoHandFrame();
  }
}

/* ============================================================
   Detection UI Update + Debounce Logic
   ============================================================ */
let lastBigLetter = null;

function processWordDetection(word, confidence, margin, landmarks) {
  const now = Date.now();
  const inCooldown = (now - state.lastCommitTime) < CONFIG.COOLDOWN_MS;
  const velocity = getAverageVelocity(landmarks);
  const motionStable = velocity <= CONFIG.MOTION_VELOCITY_THRESHOLD;
  const wordThreshold = getWordConfidenceThreshold(word);
  const hasValidPrediction = Boolean(word)
    && confidence >= wordThreshold
    && margin >= CONFIG.MIN_SCORE_MARGIN;

  if (inCooldown) {
    updateDebugStrip({ velocity, voteRatio: 0, holdProgress: 0, inCooldown: true });
    updateDetectionUI(null, 0, velocity);
    return;
  }

  if (!hasValidPrediction) {
    // Keep previous buffer while hand is present, but don't push noisy frames.
    resetStabilityTracking();
    const vote = getMajorityVote();
    updateDebugStrip({ velocity, voteRatio: vote.ratio, holdProgress: 0, inCooldown: false });
    updateDetectionUI(vote.word || null, vote.avgConfidence || 0, velocity);
    return;
  }

  if (!motionStable) {
    // Show candidate but wait for steadier motion before buffering/committing.
    const vote = getMajorityVote();
    updateDebugStrip({ velocity, voteRatio: vote.ratio, holdProgress: 0, inCooldown: false });
    updateDetectionUI(vote.word || word, Math.max(vote.avgConfidence || 0, confidence), velocity);
    return;
  }

  pushPrediction(word, confidence);
  const vote = getMajorityVote();
  const hasMajority = vote.ratio >= CONFIG.MAJORITY_RATIO;

  if (!hasMajority) {
    resetStabilityTracking();
    updateDebugStrip({ velocity, voteRatio: vote.ratio, holdProgress: 0, inCooldown });
    updateDetectionUI(vote.word, vote.avgConfidence || confidence, velocity);
    return;
  }

  if (state.stableWord !== vote.word) {
    state.stableWord = vote.word;
    state.stableStart = now;
  }

  const heldMs = now - state.stableStart;
  const holdProgress = Math.min(heldMs / CONFIG.GESTURE_HOLD_MS, 1);
  debounceBar.style.width = `${holdProgress * 100}%`;
  updateDebugStrip({ velocity, voteRatio: vote.ratio, holdProgress, inCooldown });

  updateDetectionUI(vote.word, vote.avgConfidence, velocity);

  if (heldMs >= CONFIG.GESTURE_HOLD_MS) {
    commitWord(vote.word, vote.avgConfidence);
    resetPredictionWindow();
    resetStabilityTracking();
  }
}

function finalizeOnGestureEnd() {
  const now = Date.now();
  if ((now - state.lastCommitTime) < CONFIG.COOLDOWN_MS) return;
  const vote = getMajorityVote();
  const voteThreshold = getWordConfidenceThreshold(vote.word);
  if (vote.word && vote.ratio >= CONFIG.MAJORITY_RATIO && vote.avgConfidence >= voteThreshold) {
    commitWord(vote.word, vote.avgConfidence);
  }
  resetPredictionWindow();
  resetStabilityTracking();
}

function handleNoHandFrame() {
  if (state.handPresent) {
    // Optional end-of-gesture trigger: finalize when hand leaves frame.
    finalizeOnGestureEnd();
  }
  state.handPresent = false;
  state.prevLandmarks = null;
  state.prevFrameTs = 0;
  updateDebugStrip({ velocity: 0, voteRatio: 0, holdProgress: 0, inCooldown: false });
  updateDetectionUI(null, 0, 0);
}

function updateDetectionUI(word, confidence, velocity = 0) {
  // Update detection badge
  if (word) {
    detectionBadge.style.display = 'flex';
    detectionLetter.textContent = word;
  } else {
    detectionBadge.style.display = 'none';
  }

  // Update current word
  if (word !== lastBigLetter) {
    bigLetter.textContent = word || '--';
    bigLetter.classList.remove('pop');
    void bigLetter.offsetWidth;
    bigLetter.classList.add('pop');
    lastBigLetter = word;
  }

  // Confidence bar & label
  const pct = Math.round(confidence * 100);
  confidenceBar.style.width = `${pct}%`;
  confidenceLabel.textContent = word ? `Confidence: ${pct}%` : 'Confidence: --';

  // Word description + motion state
  const stableMotion = velocity <= CONFIG.MOTION_VELOCITY_THRESHOLD;
  letterDesc.textContent = word
    ? `${WORD_DESCRIPTIONS[word] || ''}${stableMotion ? '' : ' (hold steady...)'}`
    : 'Point your hand at the camera to detect a word';

  state.currentWord   = word;
  state.currentConf   = confidence;
}

function commitWord(word, confidence) {
  const now = Date.now();
  if (!word) return;
  // Prevent repeated auto-commits while the same gesture is continuously held.
  if (state.lastCommitted === word && state.handPresent) return;
  if (state.lastCommitted === word && (now - state.lastCommitTime) < CONFIG.COOLDOWN_MS) return;

  state.word += (state.word ? ' ' : '') + word;
  state.sentence += (state.sentence ? ' ' : '') + word;
  state.letterCount++;
  state.wordCount++;
  state.confSamples.push(confidence);
  state.lastCommitted = word;
  state.lastCommitTime = now;
  renderWord();
  renderSentence();
  updateStats();

  // Visual feedback
  cameraFrame.classList.add('active');
  setTimeout(() => cameraFrame.classList.remove('active'), 300);
  showToast(`"${word}" detected`, 800);
}

/* ============================================================
   Word + Sentence Rendering
   ============================================================ */
function renderWord() {
  if (!state.word) {
    wordDisplay.innerHTML = `<span class="word-placeholder">Detected words will appear here...</span>`;
    return;
  }
  wordDisplay.innerHTML = state.word
    .split(' ')
    .filter(Boolean)
    .map((w, i) => `<span class="word-char" style="animation-delay:${i * 0.02}s">${w}</span>`)
    .join(' ');
}

function renderSentence() {
  if (!state.sentence) {
    sentenceDisplay.innerHTML = `<span class="sentence-placeholder">Complete words will build your sentence...</span>`;
    return;
  }
  sentenceDisplay.innerHTML = state.sentence
    .split(' ')
    .filter(Boolean)
    .map(w => `<span class="sentence-word">${w}</span>`)
    .join(' ');
}

/* ============================================================
   Stats
   ============================================================ */
function updateStats() {
  statLetters.textContent = state.letterCount;
  statWords.textContent   = state.wordCount;
  if (state.confSamples.length > 0) {
    const avg = state.confSamples.reduce((a,b) => a+b, 0) / state.confSamples.length;
    statAccuracy.textContent = `${Math.round(avg * 100)}%`;
  }
}

function updateSessionTime() {
  if (!state.sessionStart) return;
  const secs = Math.floor((Date.now() - state.sessionStart) / 1000);
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  statTime.textContent = `${m}:${String(s).padStart(2, '0')}`;
}

/* ============================================================
   BSL Common Words Visual Guide
   ============================================================ */
let activeBSLCategory = '';
let bslSearchQuery = '';
let bslLiveOnly = false;

function escHtml(text) {
  const s = document.createElement('span');
  s.textContent = text;
  return s.innerHTML;
}

/** Clearer step-by-step copy for live-detected signs; others fall back to `guideStepsFor`. */
const BSL_GUIDE_STEPS = {
  hello: [
    'Dominant hand flat, fingers together, palm sideways toward the viewer.',
    'Brush fingertips at your brow, then move the hand smoothly outward—small arc, not a big wave.',
  ],
  goodbye: [
    'Open flat hand (fingers together), palm facing out.',
    'Wave left–right from the wrist at about shoulder height; keep the shape readable.',
  ],
  please: [
    'Flat open hand on the centre of your chest.',
    'Rub a small circle on the chest a few times (polite, steady pressure).',
  ],
  'thank you': [
    'Flat fingertips start at your chin, palm toward your face.',
    'Move the hand forward and slightly down—like blowing a kiss without puckering.',
  ],
  sorry: [
    'Closed fist on the centre of your chest.',
    'Rub a small circle on the chest with the knuckle side (same region as please, different handshape).',
  ],
  yes: [
    'Handshape: closed fist with thumb up along the side of the index finger.',
    'Nod the fist up and down from the wrist in a clear “yes” motion.',
  ],
  no: [
    'Index and middle finger extended together; thumb, ring, and pinky tucked.',
    'Tap the pad of the index finger onto the thumb tip a couple of times.',
  ],
  help: [
    'One hand flat, palm up in front of you (the “support”).',
    'Other hand: fist with thumb pointing up; place it on the palm and lift slightly—like raising a hitch-hike thumb from a shelf.',
  ],
  eat: [
    'Tips of index and thumb pinched together (O-shape).',
    'Tap the pinch toward your closed lips a few times.',
  ],
  drink: [
    'Form a loose C with dominant hand (thumb and fingers curved).',
    'Tip the C toward your mouth as if holding a cup.',
  ],
  mother: [
    'Open flat hand, fingers together, thumb extended along the palm edge.',
    'Touch thumb pad to the side of your chin; small forward tap or hold is enough.',
  ],
  father: [
    'Same open-B handshape as mother.',
    'Touch thumb pad to mid-forehead; keep elbow relaxed so the sign stays in frame.',
  ],
  stop: [
    'Non-dominant hand flat, palm up in front of torso.',
    'Dominant flat hand chops once onto the palm (clear stop boundary).',
  ],
  want: [
    'Both hands open, fingers together, palms facing up in front of you.',
    'Pull both hands inward toward your chest a few times (whole gesture, not one hand at a time).',
  ],
  friend: [
    'Both hands: index fingers bent into hooks, other fingers tucked.',
    'Hook tips meet in front of chest; small linking or pulling motion—treat as one paired shape.',
  ],
  work: [
    'Both fists in front of torso, knuckles toward each other.',
    'Short vertical taps together like hammering—wrists stay close so the detector sees one gesture.',
  ],
  school: [
    'Both flat hands in front, palms facing each other.',
    'Bring palms together in a light clap, then separate slightly and repeat once.',
  ],
  family: [
    'Both hands in an F-shape (index and thumb extended, others curled).',
    'Hands close in front of chest; small circular motion as if drawing a ring—move both together.',
  ],
};

const TWO_HAND_BADGE_WORDS = new Set([
  'want', 'need', 'work', 'school', 'family', 'friend', 'help', 'money', 'name',
  'what', 'how', 'go', 'make', 'read', 'write', 'learn', 'love', 'happy',
  'angry', 'today', 'baby', 'cannot', 'difficult',
]);

function guideStepsFor(item) {
  if (BSL_GUIDE_STEPS[item.word]) return BSL_GUIDE_STEPS[item.word];
  const parts = item.tip.split(/\.\s+/).map(p => p.trim()).filter(Boolean);
  return parts.length > 1 ? parts : [item.tip];
}

function motionDemoClassForWord(word) {
  const map = {
    hello: 'wave-out',
    goodbye: 'wave-side',
    please: 'circle-chest',
    'thank you': 'chin-out',
    sorry: 'circle-chest',
    yes: 'nod',
    no: 'tap-pinch',
    help: 'two-help',
    eat: 'pinch-mouth',
    drink: 'c-mouth',
    mother: 'thumb-chin',
    father: 'thumb-fore',
    stop: 'chop-palm',
    want: 'two-pull',
    friend: 'two-hook',
    work: 'two-fist',
    school: 'two-clap',
    family: 'two-orbit',
  };
  return map[word] || 'generic';
}

function renderBSLGuide(query = '', category = '') {
  if (!bslGuideGrid) return;
  const q = query.trim().toLowerCase();
  let rows = BSL_COMMON_WORDS.filter(item => {
    const matchesQuery = !q || item.word.includes(q) || item.category.includes(q) || item.tip.toLowerCase().includes(q);
    const matchesCat = !category || item.category === category;
    return matchesQuery && matchesCat;
  });
  if (bslLiveOnly) rows = rows.filter(item => DETECTABLE_WORDS.has(item.word));
  // Live-detected signs surface to the top so they're easy to find while signing.
  rows.sort((a, b) => {
    const da = DETECTABLE_WORDS.has(a.word) ? 0 : 1;
    const db = DETECTABLE_WORDS.has(b.word) ? 0 : 1;
    return da - db;
  });

  const countBadge = document.getElementById('bsl-count');
  if (countBadge) countBadge.textContent = `${rows.length} sign${rows.length !== 1 ? 's' : ''}`;

  const emptyEl = document.getElementById('bsl-empty');

  if (rows.length === 0) {
    bslGuideGrid.innerHTML = '';
    if (emptyEl) emptyEl.style.display = 'flex';
    return;
  }
  if (emptyEl) emptyEl.style.display = 'none';

  bslGuideGrid.innerHTML = rows.map(item => {
    const isDetectable = DETECTABLE_WORDS.has(item.word);
    const steps = guideStepsFor(item);
    const demo = motionDemoClassForWord(item.word);
    const scope = TWO_HAND_BADGE_WORDS.has(item.word)
      ? '<span class="bsl-hand-scope">Often two hands or uses the body</span>'
      : '';
    const hasExtraSteps = steps.length > 0
      && !(steps.length === 1 && steps[0].trim() === item.tip.trim());
    const stepList = steps.map(s => `<li>${escHtml(s)}</li>`).join('');
    const stepsBlock = hasExtraSteps
      ? `<details class="bsl-steps-toggle"><summary>How to sign</summary><ol class="bsl-steps">${stepList}</ol></details>`
      : '';
    return `
      <article class="bsl-word-card${isDetectable ? ' detectable' : ''}">
        <div class="bsl-word-head">
          <div class="bsl-word-name">${escHtml(item.word)}${isDetectable ? '<span class="live-pill">live</span>' : ''}</div>
          <div class="bsl-word-cat">${escHtml(item.category)}</div>
        </div>
        <div class="bsl-motion-demo bsl-motion-demo--${demo}" aria-hidden="true" title="Motion hint"></div>
        ${scope ? `<div class="bsl-scope-row">${scope}</div>` : ''}
        <p class="bsl-word-tip">${escHtml(item.tip)}</p>
        ${stepsBlock}
      </article>
    `;
  }).join('');
}

/* ============================================================
   Camera Setup
   ============================================================ */
async function startCamera() {
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: state.facingMode,
        width: { ideal: 960 },
        height: { ideal: 540 },
      },
      audio: false,
    });

    video.srcObject = state.stream;
    await video.play();

    // Size canvas
    video.addEventListener('loadedmetadata', () => {
      canvas.width  = video.videoWidth  || 960;
      canvas.height = video.videoHeight || 540;
    }, { once: true });

    // Show camera UI
    placeholder.style.display    = 'none';
    video.style.display          = 'block';
    canvas.style.display         = 'block';
    cameraHud.style.display      = 'flex';
    cameraCtrl.style.display     = 'flex';
    detectionBadge.style.display = 'none';
    state.cameraActive = true;

    // Session timer
    state.sessionStart = Date.now();
    state.sessionTimer = setInterval(updateSessionTime, 1000);

    // Init MediaPipe
    initMediaPipe();

    showToast('Camera started — show your hand!');
  } catch (err) {
    console.error('Camera error:', err);
    showToast('Camera access denied. Please allow camera permission.');
  }
}

function stopCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach(t => t.stop());
    state.stream = null;
  }
  if (state.tracker) {
    state.tracker.close();
    state.tracker = null;
  }
  if (state.mpCamera) {
    state.mpCamera.stop();
    state.mpCamera = null;
  }
  if (state.animFrame) {
    cancelAnimationFrame(state.animFrame);
    state.animFrame = null;
  }
  clearInterval(state.sessionTimer);
  clearInterval(state.fpsInterval);

  video.srcObject = null;
  video.style.display          = 'none';
  canvas.style.display         = 'none';
  cameraHud.style.display      = 'none';
  cameraCtrl.style.display     = 'none';
  detectionBadge.style.display = 'none';
  placeholder.style.display    = 'flex';
  cameraFrame.classList.remove('active');

  state.cameraActive = false;
  resetPredictionWindow();
  resetStabilityTracking();
  state.prevLandmarks = null;
  state.prevFrameTs = 0;
  state.handPresent = false;
  updateDebugStrip({ velocity: 0, voteRatio: 0, holdProgress: 0, inCooldown: false });
  hudHandInfo.textContent       = 'Scanning for hand...';
  hudFps.textContent            = '-- fps';
  updateDetectionUI(null, 0);
  showToast('Camera stopped');
}

async function flipCamera() {
  state.facingMode = state.facingMode === 'user' ? 'environment' : 'user';
  await stopCameraOnly();
  await startCamera();
}

async function stopCameraOnly() {
  if (state.stream) {
    state.stream.getTracks().forEach(t => t.stop());
    state.stream = null;
  }
  if (state.tracker) {
    state.tracker.close();
    state.tracker = null;
  }
  if (state.mpCamera) {
    state.mpCamera.stop();
    state.mpCamera = null;
  }
}

/* ============================================================
   MediaPipe Hands
   ============================================================ */
function initMediaPipe() {
  const tracker = new Holistic({
    locateFile: file =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
  });

  tracker.setOptions({
    modelComplexity:          0,
    smoothLandmarks:          true,
    refineFaceLandmarks:      false,
    minDetectionConfidence:   0.55,
    minTrackingConfidence:    CONFIG.MIN_TRACKING_CONFIDENCE,
  });

  tracker.onResults(results => {
    if (!canvas.width) {
      canvas.width  = video.videoWidth  || 960;
      canvas.height = video.videoHeight || 540;
    }
    renderFrame(results);
    trackFPS();
  });

  state.tracker = tracker;

  // Use MediaPipe Camera utils to drive frame capture
  const mpCamera = new Camera(video, {
    onFrame: async () => {
      if (state.tracker) {
        await state.tracker.send({ image: video });
      }
    },
    width: 960,
    height: 540,
  });
  state.mpCamera = mpCamera;
  mpCamera.start();

  // FPS tracking
  clearInterval(state.fpsInterval);
  state.fpsInterval = setInterval(() => {
    hudFps.textContent = `${state.fps} fps`;
    state.fps = 0;
  }, 1000);
}

function trackFPS() {
  state.fps++;
}

/* ============================================================
   Controls
   ============================================================ */
$('btn-start-camera').addEventListener('click', startCamera);
$('btn-stop-camera').addEventListener('click', stopCamera);

$('btn-flip-camera').addEventListener('click', flipCamera);

$('btn-toggle-landmarks').addEventListener('click', function() {
  state.showLandmarks = !state.showLandmarks;
  this.dataset.active = state.showLandmarks;
  showToast(state.showLandmarks ? 'Landmarks visible' : 'Landmarks hidden');
});

$('btn-clear-word').addEventListener('click', () => {
  state.word = '';
  state.sentence = '';
  state.lastCommitted = '';
  state.lastCommitTime = 0;
  resetPredictionWindow();
  resetStabilityTracking();
  updateDebugStrip({ velocity: 0, voteRatio: 0, holdProgress: 0, inCooldown: false });
  renderWord();
  renderSentence();
  showToast('Detected words cleared');
});

$('btn-speak').addEventListener('click', () => {
  if (!state.sentence) { showToast('Nothing to speak'); return; }
  const utt = new SpeechSynthesisUtterance(state.sentence);
  utt.rate  = 0.9;
  utt.pitch = 1;
  speechSynthesis.cancel();
  speechSynthesis.speak(utt);
  showToast('Speaking...');
});

$('btn-clear-sentence').addEventListener('click', () => {
  state.sentence = '';
  renderSentence();
  showToast('Sentence cleared');
});

$('btn-reset-stats').addEventListener('click', () => {
  state.letterCount  = 0;
  state.wordCount    = 0;
  state.confSamples  = [];
  state.sessionStart = state.cameraActive ? Date.now() : null;
  updateStats();
  statAccuracy.textContent = '--%';
  statTime.textContent     = '0:00';
  showToast('Stats reset');
});

// Modals
$('btn-how-it-works').addEventListener('click', () => {
  $('modal-hiw').style.display = 'flex';
});
$('btn-privacy').addEventListener('click', () => {
  $('modal-privacy').style.display = 'flex';
});
document.querySelectorAll('.modal-close').forEach(btn => {
  btn.addEventListener('click', () => {
    $(`${btn.dataset.modal}`).style.display = 'none';
  });
});
document.querySelectorAll('.modal-overlay').forEach(overlay => {
  overlay.addEventListener('click', e => {
    if (e.target === overlay) overlay.style.display = 'none';
  });
});

/* ============================================================
   Toast Notification
   ============================================================ */
let toastTimer = null;
function showToast(msg, duration = 2200) {
  clearTimeout(toastTimer);
  toast.textContent = msg;
  toast.classList.add('show');
  toastTimer = setTimeout(() => toast.classList.remove('show'), duration);
}

/* ============================================================
   Tab Switching
   ============================================================ */
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    const pane = document.getElementById(`tab-${tab}`);
    if (pane) pane.classList.add('active');
  });
});

/* ============================================================
   Category Chip Filtering
   ============================================================ */
document.querySelectorAll('.cat-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    document.querySelectorAll('.cat-chip').forEach(c => c.classList.remove('active'));
    chip.classList.add('active');
    activeBSLCategory = chip.dataset.cat;
    renderBSLGuide(bslSearchQuery, activeBSLCategory);
  });
});

/* ============================================================
   BSL Search with clear button
   ============================================================ */
const bslClearBtn = document.getElementById('bsl-search-clear');
if (bslSearch) {
  bslSearch.addEventListener('input', e => {
    bslSearchQuery = e.target.value || '';
    if (bslClearBtn) bslClearBtn.style.display = bslSearchQuery ? 'block' : 'none';
    renderBSLGuide(bslSearchQuery, activeBSLCategory);
  });
}
const cheatToggle = document.getElementById('cheat-toggle');
const cheatBody = document.getElementById('cheat-body');
if (cheatToggle && cheatBody) {
  cheatToggle.addEventListener('click', () => {
    const open = cheatToggle.getAttribute('aria-expanded') !== 'false';
    cheatToggle.setAttribute('aria-expanded', open ? 'false' : 'true');
    cheatBody.classList.toggle('collapsed', open);
  });
}

/* Cheatsheet rows: each live-detected sign with its zone, motion preview animation
 * (reuses the BSL guide CSS animations), and the precise detector contract.
 */
const CHEAT_ROWS = [
  { word: 'hello',     zone: 'FOREHEAD', zoneClass: 'z-face',  what: 'Open palm at brow.',         how: 'Fingertips brush forehead, thumb at the side.' },
  { word: 'father',    zone: 'FOREHEAD', zoneClass: 'z-face',  what: 'Thumb pad on forehead.',     how: 'Thumb touches forehead, other fingers extend upward.' },
  { word: 'thank you', zone: 'CHIN',     zoneClass: 'z-chin',  what: 'Fingertips at chin, move forward.', how: 'Fingertips touch chin, palm faces you, then arc out.' },
  { word: 'mother',    zone: 'CHIN',     zoneClass: 'z-chin',  what: 'Thumb pad on chin.',         how: 'Thumb touches chin, fingers extend up/out.' },
  { word: 'goodbye',   zone: 'SHOULDER', zoneClass: 'z-mid',   what: 'Open palm waving to side.',  how: 'Hand at shoulder height, away from face/chest, wave wrist.' },
  { word: 'please',    zone: 'CHEST',    zoneClass: 'z-chest', what: 'Open palm, small circle on chest.', how: 'Wrist clearly at chest level (below the chin).' },
  { word: 'sorry',     zone: 'CHEST',    zoneClass: 'z-chest', what: 'Closed fist, small circle on chest.', how: 'Fingers fully curled, fist at chest.' },
  { word: 'eat',       zone: 'MOUTH',    zoneClass: 'z-mouth', what: 'Bunched fingertips tap mouth.', how: 'Thumb + index pinched together, near mouth.' },
  { word: 'drink',     zone: 'MOUTH',    zoneClass: 'z-mouth', what: 'C-hand tips to mouth.',      how: 'Thumb + index + middle out, ring & pinky curled, near mouth.' },
  { word: 'yes',       zone: 'HANDSHAPE',zoneClass: 'z-hand',  what: 'Fist nods up & down.',       how: 'All fingers curled, thumb up along the fist.' },
  { word: 'no',        zone: 'HANDSHAPE',zoneClass: 'z-hand',  what: 'Index + middle tap thumb.',  how: 'Only index & middle extended, tips meet thumb tip.' },
  { word: 'help',      zone: 'TWO HANDS',zoneClass: 'z-two',   what: 'Thumbs-up fist lifts from flat palm.', how: 'One curled hand + thumbs-up, both hands near each other.' },
  { word: 'stop',      zone: 'TWO HANDS',zoneClass: 'z-two',   what: 'Flat palm chops onto flat palm.', how: 'Both palms open, one above the other, x-aligned.' },
  { word: 'school',    zone: 'TWO HANDS',zoneClass: 'z-two',   what: 'Two flat palms clap side-by-side.', how: 'Both palms open, wrists at similar height, close together.' },
  { word: 'want',      zone: 'TWO HANDS',zoneClass: 'z-two',   what: 'Both open palms pull toward chest.', how: 'Both hands open, below the chin, fairly close together.' },
  { word: 'friend',    zone: 'TWO HANDS',zoneClass: 'z-two',   what: 'Hooked index fingers link.', how: 'Both hands curled, wrists very close together.' },
  { word: 'work',      zone: 'TWO HANDS',zoneClass: 'z-two',   what: 'Fists tap together.',        how: 'Both fists curled, at mid-body, close together.' },
  { word: 'family',    zone: 'TWO HANDS',zoneClass: 'z-two',   what: 'Two open hands circle together.', how: 'Both palms open, mid/upper body, close together.' },
];

function renderCheatsheet() {
  const grid = document.getElementById('cheat-grid');
  if (!grid) return;
  grid.innerHTML = CHEAT_ROWS.map(row => {
    const demo = motionDemoClassForWord(row.word);
    return `
      <article class="cheat-row">
        <div class="cheat-row-head">
          <span class="cheat-zone ${row.zoneClass}">${row.zone}</span>
          <strong class="cheat-word">${escHtml(row.word)}</strong>
        </div>
        <div class="cheat-row-demo bsl-motion-demo bsl-motion-demo--${demo}" aria-hidden="true"></div>
        <div class="cheat-row-text">
          <p class="cheat-what">${escHtml(row.what)}</p>
          <p class="cheat-how"><span class="cheat-how-label">Detector:</span> ${escHtml(row.how)}</p>
        </div>
      </article>
    `;
  }).join('');
}

renderCheatsheet();

const bslLiveOnlyEl = document.getElementById('bsl-live-only');
if (bslLiveOnlyEl) {
  bslLiveOnlyEl.addEventListener('change', e => {
    bslLiveOnly = !!e.target.checked;
    renderBSLGuide(bslSearchQuery, activeBSLCategory);
  });
}
if (bslClearBtn) {
  bslClearBtn.addEventListener('click', () => {
    bslSearch.value = '';
    bslSearchQuery = '';
    bslClearBtn.style.display = 'none';
    renderBSLGuide('', activeBSLCategory);
    bslSearch.focus();
  });
}

/* ============================================================
   Detectable Signs Chips
   ============================================================ */
function renderDetectableChips() {
  const container = document.getElementById('detectable-chips');
  if (!container) return;
  const words = [...DETECTABLE_WORDS].sort();
  const chipCountEl = document.getElementById('detectable-chip-count');
  if (chipCountEl) chipCountEl.textContent = `${words.length} words`;
  container.innerHTML = words.map(w => `
    <div class="detect-chip">
      <span class="detect-chip-dot"></span>
      ${w}
    </div>
  `).join('');
}

/* ============================================================
   Init
   ============================================================ */
renderWord();
renderSentence();
renderBSLGuide('', '');
renderDetectableChips();
updateDebugStrip({ velocity: 0, voteRatio: 0, holdProgress: 0, inCooldown: false });

// Word log clear on panel button (alias)
const btnClearWordPanel = document.getElementById('btn-clear-word-panel');
if (btnClearWordPanel) {
  btnClearWordPanel.addEventListener('click', () => {
    state.word = '';
    state.sentence = '';
    state.lastCommitted = '';
    state.lastCommitTime = 0;
    resetPredictionWindow();
    resetStabilityTracking();
    updateDebugStrip({ velocity: 0, voteRatio: 0, holdProgress: 0, inCooldown: false });
    renderWord();
    renderSentence();
    showToast('Cleared');
  });
}

// Canvas roundRect polyfill
if (!CanvasRenderingContext2D.prototype.roundRect) {
  CanvasRenderingContext2D.prototype.roundRect = function(x, y, w, h, r) {
    if (w < 2 * r) r = w / 2;
    if (h < 2 * r) r = h / 2;
    this.beginPath();
    this.moveTo(x + r, y);
    this.arcTo(x + w, y, x + w, y + h, r);
    this.arcTo(x + w, y + h, x, y + h, r);
    this.arcTo(x, y + h, x, y, r);
    this.arcTo(x, y, x + w, y, r);
    this.closePath();
    return this;
  };
}

console.log('%cSignSpeak Engine ready', 'color:#6D28D9;font-weight:700;font-size:14px');
