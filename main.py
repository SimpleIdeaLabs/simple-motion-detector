import cv2
from numpy import diff
import simpleaudio


def main():
    camera = cv2.VideoCapture(0)

    while camera.isOpened():
        ret, frame1 = camera.read()
        ret, frame2 = camera.read()
        contours = get_refined_contours(frame1=frame1, frame2=frame2)
        detect_movements(display_frame=frame1, contours=contours)
        if cv2.waitKey(10) == ord('q'):
            break
        cv2.imshow("My Camera", frame1)


def get_refined_contours(**kwargs):
    frame1 = kwargs.get('frame1')
    frame2 = kwargs.get('frame2')
    movement = cv2.absdiff(frame1, frame2)
    grayed_movement = cv2.cvtColor(movement, cv2.COLOR_RGB2GRAY)
    blurred_movement = cv2.GaussianBlur(grayed_movement, (5, 5), 0)
    _, threshold_movement = cv2.threshold(
        blurred_movement, 20, 255, cv2.THRESH_BINARY)
    dilated_movement = cv2.dilate(threshold_movement, None, iterations=3)
    contours, _ = cv2.findContours(
        dilated_movement, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def detect_movements(**kwargs):
    display_frame = kwargs.get('display_frame')
    contours = kwargs.get('contours')
    for c in contours:
        if cv2.contourArea(c) < 20000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        play_beep_sound()


def play_beep_sound():
    wave_obj = simpleaudio.WaveObject.from_wave_file("beep.wav")
    wave_obj.play()


if __name__ == "__main__":
    main()
