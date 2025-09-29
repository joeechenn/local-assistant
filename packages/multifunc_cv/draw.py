import cv2

TEXT_SPACING = 10
FONT_SIZE = 1
FONT_THICKNESS = 2

RADIUS = 3

def draw_face(img, faces, color=(0, 0, 225)):
    for f in faces:
        x,y,w,h = f['bbox']
        start_point = (x, y)
        end_point = (x + w, y + h)
        cv2.rectangle(img, start_point, end_point, color, 2)

        if f["score"] is not None:
            (text_w, text_h), _ = cv2.getTextSize(
                f"{f['score']:.2f}",
                cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE,
                FONT_THICKNESS,
            )
            cv2.putText(img, f"{f['score']:.2f}",
                        (x + w // 2 - text_w // 2,
                         y + h + TEXT_SPACING + text_h),
                        cv2.FONT_HERSHEY_PLAIN,
                        FONT_SIZE,
                        color,
                        FONT_THICKNESS)

def draw_hands(img, hands, color=(255,0,0)):
    for h in hands:
        pts = h['landmarks']
        for p in pts:
            cv2.circle(img, p, RADIUS, color, -1)
        if h["gesture"]:
            x0, y0 = pts[0]
            cv2.putText(img,
                        f"{h['gesture']} ({h['score']:.2f})",
                        (x0, y0 - TEXT_SPACING),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        FONT_SIZE, color,
                        FONT_THICKNESS)