import streamlit as st
import sys
import numpy as np
import cv2 as cv
from ultralytics import YOLO                # to use obj detection model
from streamlit.web.cli import main
from streamlit.web import cli as stcli
from PIL import Image
import tempfile                             # for video preprocessing
import av
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PAGE LAYOUT
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
st.set_page_config('objDetection', layout='wide')


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# CREDIT SECTION
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

st.markdown(
    """
    <div style='display: flex; justify-content: flex-end; margin-bottom: 10px;'>
        <div style='display: inline-flex; align-items: center; background-color: rgba(255, 255, 255, 0.05); 
                    padding: 6px 14px; border-radius: 25px; border: 1px solid rgba(255, 255, 255, 0.1);'>
            <b style='margin-right: 12px; font-size: 15px;'>Credits :</b>
            <a href='https://github.com/pradhans369' target='_blank'>
                <img src='https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white' 
                     style='border-radius: 6px; margin-right: 8px;'>
            </a>
            <a href='https://www.linkedin.com/in/pradhans369/' target='_blank'>
                <img src='https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white' 
                     style='border-radius: 6px;'>
            </a>
        </div>
    </div>
    """, 
    unsafe_allow_html=True
)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# HEADINGS
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

st.title("Project Vision | Object Detection")
st.write("Detecting Over More Than 100 Classes")
st.markdown("---")

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# MODEL
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@st.cache_resource
def load_model():
    return YOLO("yolo26n.pt")

model = load_model()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# MAIN BUTTON SECTION
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

btn_sec1, btn_sec2 = st.columns([1,1])

with btn_sec1:
    button = st.selectbox('Select Input Source', options=['Image', 'Recorded Video', 'Live webcam'], index=None)

st.divider()

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 'IMAGE' BUTTON
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

if button == 'Image':
    with btn_sec2:
        img_btn = st.selectbox('Select Image Input', options=['Open Camera', 'Upload image'], index=None)


    img_format = ['.jpg','.jpeg','.png','.gif','.webp']
    btn_sec1, btn_sec2 = st.columns([1, 1])

    if img_btn == 'Open Camera':
        with btn_sec1:
            img_clicked = st.camera_input("Take live picture")                              # used for clicking image

        if img_clicked is not None:
            img = Image.open(img_clicked)                                                   # converting image buffer to opencv format

                # Converts Raw Bytes to a PIL Image: Uses Python's Pillow library (PIL.Image) to read and decode the raw byte stream from 'img_clicked' into a workable PIL Image object.
                # Prepares for Processing: Once it is a PIL image, you can easily convert it to a NumPy array (np.array(img)), pass it into computer vision models (like YOLO), or perform transformations (resizing, filtering, etc.).

            img_array = np.array(img)           # converting the img into a numpy array

            # sending the numpy array into the model
            result = model(img_array)

            with btn_sec2:
                st.subheader("Detected Objects in the Image")
                st.image(result[0].plot(), caption='detected objects')
        else:
            with btn_sec2:
                st.info("Snap a photo on the left to see detections here")

    elif img_btn == 'Upload image':
        with btn_sec1:
            st.write("Select image from files")
            imgs_uploaded = st.file_uploader('Select image / images', type=img_format, accept_multiple_files=True)         # accepting multiple image files

            if imgs_uploaded:
                with btn_sec2:
                    st.subheader("Detected Objects in the Image")

                for img in imgs_uploaded:

                    img = Image.open(img)
                    img_array = np.array(img)

                    result = model(img_array)

                    with btn_sec2:
                        st.image(result[0].plot())
            else:
                with btn_sec2:
                    st.info("Select a photo from the files to see detections here")


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 'RECORDED VIDEO' BUTTON
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


elif button == 'Recorded Video':
    vid_format = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    btn_sec1, btn_sec2 = st.columns([1, 1])


    with btn_sec1:
        rec_vid = st.file_uploader('Upload Video (one video only)', type=vid_format)
        

    with btn_sec2:
        if rec_vid is not None:
            # saving the uploaded video bytes to a file on a disk with mp4 format
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(rec_vid.read())
            tfile.close()                   # closing the file so 'OpenCV' can read it

            col1, col2 = st.columns([1,1])
            with col1:
                generate = st.button('Generate')

            if generate:
                cap = cv.VideoCapture(tfile.name)

                width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) or 100
                fps = int(cap.get(cv.CAP_PROP_FPS)) or 30

                out_tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                fourcc = cv.VideoWriter_fourcc(*'mp4v')
                out = cv.VideoWriter(out_tfile.name, fourcc, fps, (width, height))

                # making a progressbar for the download button 
                progress_bar = st.progress(0, text='Processing the video to download')
                vid_play = st.empty()
                current_frame = 0
                
                while cap.isOpened():
                    ret, frames = cap.read()
                    if not ret:
                        print('ERROR')
                        break

                    results = model(frames)
                    out.write(results[0].plot())

                    rgb_vid = cv.cvtColor(results[0].plot(), cv.COLOR_BGR2RGB)
                    vid_play.image(rgb_vid, channels='RGB')

                    # updating progressbar by each frame processed
                    current_frame += 1
                    progress = min(current_frame / total_frames, 1.0)
                    progress_bar.progress(
                        progress,
                        text=f"Processing download : frame {current_frame} of {total_frames} ({int(progress * 100)} %)"
                    )


                cap.release()
                out.release()
                st.toast('Video processing complete !!!')

                with col2:
                    # making a download button for the generated video
                    with open(out_tfile.name, 'rb') as f:
                        video_bytes = f.read()

                    st.download_button(
                        label='Download',
                        data=video_bytes,
                        file_name="yolo_detected_video.mp4",
                        mime="video/mp4"
                    )
        else:
            st.info('Upload a video file to see object detection')



# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# 'LIVE WEB CAM' BUTTON
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# THIS SECTION OF THE CODE IS WRITTEN WITH THE HELP OF AI,
# coz open cv can work properly on local host, but while deploying to the server, it throws error.
# Hence the live web cam section of the web app, is written using HTML, CSS, & JS. 


elif button == 'Live webcam':
    st.subheader("Live Webcam Object Detection")
    st.caption("Real-time YOLO detection in HD quality.")

    # STUN server configuration for cloud deployment
    RTC_CONFIGURATION = RTCConfiguration(
        {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
            ]
        }
    )

    class YOLOVideoProcessor:
        def __init__(self):
            self.frame_count = 0
            self.last_results = None

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            self.frame_count += 1

            # Process YOLO on alternating frames to keep CPU load low and video quality in HD
            if self.frame_count % 2 == 0 or self.last_results is None:
                self.last_results = model(img, imgsz=640)

            # Draw sharp, crisp YOLO bounding boxes and labels
            annotated_frame = self.last_results[0].plot(
                line_width=2,
                font_size=12,
                labels=True,
                conf=True
            )

            return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    webrtc_streamer(
        key="yolo-live-webcam-hd",
        video_processor_factory=YOLOVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {
                "width": {"min": 1280, "ideal": 1280},
                "height": {"min": 720, "ideal": 720},
                "frameRate": {"ideal": 30, "min": 15},
            },
            "audio": False,
        },
        async_processing=True,
    )

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    if st.runtime.exists():
        pass
    else:
        sys.argv = ['streamlit', 'run', sys.argv[0]]
        sys.exit(stcli.main())

