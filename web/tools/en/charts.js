  var N1 = [
    { k: "A. Muse Spark alone", v: [0.074, 0.273, 0.821] },
    { k: "A plus glove learned shift", v: [0.195, 0.376, 0.850], hi: true },
    { k: "B. SAM rule", v: [0.090, 0.214, 0.667] },
    { k: "C. B plus Muse Spark check", v: [0.112, 0.227, 0.679] },
    { k: "C plus glove learned shift", v: [0.137, 0.266, 0.690] },
    { k: "Reference: always in contact", v: [0.042, 0.042, 0.600], ref: true }
  ];
  var N1M = ["Boundary F1 ±5 frames", "Boundary F1 ±15 frames", "Contact IoU"];
  var N2 = [
    { g: "Known objects (1,426 held-out)" },
    { k: "Ask Muse Spark in one go", a: 0.501, lo: 0.469, hi: 0.529, c: "base", note: "One question: \"can a robot hand copy this\"" },
    { k: "Checklist, equal weights", a: 0.617, lo: 0.579, hi: 0.659, c: "hi", note: "Plain average of the 7 Muse Spark questions" },
    { k: "Checklist, learned weights", a: 0.622, lo: 0.588, hi: 0.661, c: "hi", note: "The same 7 with logistic regression" },
    { k: "Our score (12 questions)", a: 0.624, lo: 0.589, hi: 0.661, c: "hi", b: true, note: "Muse Spark 7, SAM 2, computed from video 3" },
    { k: "Object name only (tuning rate)", a: 0.771, lo: 0.744, hi: 0.792, c: "ref", note: "Success rate of each object measured on tuning. Works only for objects that have labels" },
    { g: "Unseen objects (learned again without that object)" },
    { k: "Our score", a: 0.571, lo: 0.534, hi: 0.609, c: "hi", b: true, note: "Still separates clips on objects not in training" },
    { k: "Object name only", a: 0.500, c: "ref", note: "An unseen object has no rate, so it cannot decide (0.5 by definition)" }
  ];
  var CKL = [
    { s: "Muse Spark", k: "Shape", q: "Middle side / longest side", r: "0.6 or more, chunky", w: 1.252 },
    { s: "Muse Spark", k: "Wrap width", q: "The middle side the hand must wrap", r: "8 cm or less", w: 0.922 },
    { s: "Muse Spark", k: "Finger occlusion", q: "Are the fingertips visible during the grasp", r: "Most fingertips visible", w: 0.481 },
    { s: "Video", k: "Hand shape", q: "Logistic regression on 10 MediaPipe hand features", r: "Top third on tuning", w: 0.434 },
    { s: "Muse Spark", k: "Size vs hand", q: "Longest side of the object", r: "15 cm or less", w: 0.391 },
    { s: "Muse Spark", k: "Handle", q: "Is there a part made to be held", r: "Yes", w: 0.258 },
    { s: "SAM", k: "Contact clarity", q: "Is the hand next to the object only during contact", r: "Top third on tuning", w: 0.237 },
    { s: "SAM", k: "Visibility", q: "Share of frames where hand and object are both visible", r: "Top third on tuning", w: 0.043 },
    { s: "Video", k: "Camera stability", q: "Camera motion during the clip", r: "Top third on tuning", w: -0.030 },
    { s: "Muse Spark", k: "Free space", q: "Gap between the grasp point and the nearest other object", r: "10 cm or more", w: -0.171 },
    { s: "Video", k: "Motion completeness", q: "Are the reach and the release in the video", r: "Both visible", w: -0.331 },
    { s: "Muse Spark", k: "Thickness", q: "Shortest side", r: "4 cm or more", w: -0.342 }
  ];

