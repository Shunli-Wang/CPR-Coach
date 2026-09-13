#!/usr/bin/env python3
"""
CPR Arm-Placement Coach -- live webcam feedback (YOLO-pose edition).

Uses a multi-person YOLO pose model so a lifelike CPR MANIKIN can't hijack the
detection. Every frame we find all people, keep the most UPRIGHT torso (that's
the kneeling rescuer -- shoulders well above hips) and ignore the horizontal
one (the manikin lying down). Then we score the three things that matter most
for good chest compressions:

  1. Elbows LOCKED straight   (shoulder-elbow-wrist angle near 180 deg)
  2. Arms VERTICAL            (wrist roughly under the shoulder, not leaning)
  3. Shoulders squared        (both shoulders level over the hands)

This is a practice aid, NOT a medical device. It is completely separate from
the CPR-Coach research code in this repo.

Setup for best results:
  - Place the laptop to your SIDE (side-on view) so it can see your arm line.
  - Kneel next to your manikin with hands on the chest.
  - Make sure your shoulder, elbow and wrist are all in frame.

Controls:  press  q  or  Esc  to quit.

Run:
  ./.cpr-webcam-venv/bin/python cpr_arm_coach.py
"""

import math
import sys

import cv2
from ultralytics import YOLO

MODEL_PATH = "yolo11n-pose.pt"   # auto-downloads if missing

# COCO-17 keypoint indices
NOSE = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12

# --- feedback thresholds (tune to taste) ---
ELBOW_STRAIGHT_MIN = 160.0   # deg; >= this counts as "locked straight"
ARM_VERTICAL_MAX = 25.0      # deg from vertical; <= this counts as "over the hands"
SHOULDER_LEVEL_MAX = 12.0    # deg tilt between shoulders; <= this counts as "square"
KP_CONF_MIN = 0.5            # min keypoint confidence to trust a joint

GREEN = (0, 200, 0)
RED = (0, 0, 255)
AMBER = (0, 170, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
CYAN = (255, 255, 0)


def angle_deg(a, b, c):
    """Angle at vertex b formed by points a-b-c, in degrees (0..180)."""
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    dot = bax * bcx + bay * bcy
    na = math.hypot(bax, bay)
    nc = math.hypot(bcx, bcy)
    if na == 0 or nc == 0:
        return 0.0
    cosv = max(-1.0, min(1.0, dot / (na * nc)))
    return math.degrees(math.acos(cosv))


def vertical_deviation(shoulder, wrist):
    """Angle (deg) of the shoulder->wrist line away from straight-down vertical."""
    dx = wrist[0] - shoulder[0]
    dy = wrist[1] - shoulder[1]
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def draw_label(img, text, org, color, scale=0.7, thick=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(img, (x - 6, y - th - 8), (x + tw + 6, y + 8), BLACK, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thick, cv2.LINE_AA)


def uprightness(kp, conf):
    """Score how upright a person is: shoulders well ABOVE hips (kneeling
    rescuer) scores high; a body lying flat (the manikin) scores near zero or
    negative. Returns None if the torso keypoints aren't trustworthy."""
    need = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
    if min(conf[i] for i in need) < 0.3:
        return None
    sy = (kp[L_SHOULDER][1] + kp[R_SHOULDER][1]) / 2
    hy = (kp[L_HIP][1] + kp[R_HIP][1]) / 2
    sx = (kp[L_SHOULDER][0] + kp[R_SHOULDER][0]) / 2
    hx = (kp[L_HIP][0] + kp[R_HIP][0]) / 2
    return abs(sy - hy) - abs(sx - hx)


def pick_rescuer(kps, confs):
    """From all detected people, return (kp, conf) for the most upright torso.
    Falls back to the highest-confidence person if no torso is reliable."""
    best_i, best_score = None, -1e9
    for i, (kp, cf) in enumerate(zip(kps, confs)):
        s = uprightness(kp, cf)
        if s is not None and s > best_score:
            best_score, best_i = s, i
    if best_i is None:                       # no clean torso -> most confident
        best_i = max(range(len(kps)), key=lambda i: float(confs[i].mean()))
    return kps[best_i], confs[best_i]


def pick_arm(conf):
    """Return the more-confident arm's (shoulder, elbow, wrist) indices."""
    left = (L_SHOULDER, L_ELBOW, L_WRIST)
    right = (R_SHOULDER, R_ELBOW, R_WRIST)
    return left if min(conf[i] for i in left) >= min(conf[i] for i in right) else right


def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam (index 0).", file=sys.stderr)
        print("macOS: grant camera permission to your terminal in "
              "System Settings > Privacy & Security > Camera.", file=sys.stderr)
        sys.exit(1)

    win = "CPR Arm-Placement Coach  (press q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("ERROR: failed to read frame from webcam.", file=sys.stderr)
            break

        frame = cv2.flip(frame, 1)  # mirror -> feels natural
        h, w = frame.shape[:2]

        res = model(frame, verbose=False)[0]
        checks = []  # (label, ok_bool, detail)

        has_people = (res.keypoints is not None and len(res.keypoints) > 0
                      and res.keypoints.conf is not None)
        if has_people:
            kps = res.keypoints.xy.cpu().numpy()      # (n,17,2)
            confs = res.keypoints.conf.cpu().numpy()  # (n,17)
            kp, cf = pick_rescuer(kps, confs)

            s_i, e_i, wr_i = pick_arm(cf)
            arm_ok_seen = min(cf[s_i], cf[e_i], cf[wr_i]) >= KP_CONF_MIN

            if arm_ok_seen:
                sh = tuple(map(int, kp[s_i]))
                el = tuple(map(int, kp[e_i]))
                wr = tuple(map(int, kp[wr_i]))

                elbow_ang = angle_deg(sh, el, wr)
                elbow_ok = elbow_ang >= ELBOW_STRAIGHT_MIN
                checks.append(("Elbow locked", elbow_ok, f"{elbow_ang:0.0f} deg"))

                vdev = vertical_deviation(sh, wr)
                vert_ok = vdev <= ARM_VERTICAL_MAX
                checks.append(("Arm vertical", vert_ok, f"{vdev:0.0f} deg off"))

                shoulders_seen = min(cf[L_SHOULDER], cf[R_SHOULDER]) >= KP_CONF_MIN
                if shoulders_seen:
                    ls = tuple(map(int, kp[L_SHOULDER]))
                    rs = tuple(map(int, kp[R_SHOULDER]))
                    tilt = math.degrees(math.atan2(abs(ls[1] - rs[1]),
                                                   max(1, abs(ls[0] - rs[0]))))
                    sh_ok = tilt <= SHOULDER_LEVEL_MAX
                    checks.append(("Shoulders square", sh_ok, f"{tilt:0.0f} deg tilt"))

                # cyan = ideal vertical from the shoulder
                cv2.line(frame, (sh[0], sh[1]), (sh[0], wr[1]), CYAN, 2, cv2.LINE_AA)
                col = GREEN if (elbow_ok and vert_ok) else RED
                cv2.line(frame, sh, el, col, 5, cv2.LINE_AA)
                cv2.line(frame, el, wr, col, 5, cv2.LINE_AA)
                for p in (sh, el, wr):
                    cv2.circle(frame, p, 9, col, -1)
                    cv2.circle(frame, p, 9, WHITE, 2)
                draw_label(frame, f"elbow {elbow_ang:0.0f}", (el[0] + 12, el[1]),
                           WHITE, 0.7, 2)
            else:
                draw_label(frame, "Arm not clearly visible - turn side-on",
                           (20, h - 30), AMBER)
        else:
            draw_label(frame, "No person detected - step into frame (side-on)",
                       (20, h - 30), AMBER)

        # feedback panel
        y = 44
        all_ok = bool(checks) and all(ok for _, ok, _ in checks)
        header = "GOOD FORM" if all_ok else ("ADJUST" if checks else "WAITING")
        draw_label(frame, header, (20, y), GREEN if all_ok else AMBER, 0.95, 2)
        y += 44
        for label, ok, detail in checks:
            mark = "OK  " if ok else "X   "
            draw_label(frame, f"{mark}{label}: {detail}", (20, y),
                       GREEN if ok else RED)
            y += 34

        cv2.imshow(win, frame)
        if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
