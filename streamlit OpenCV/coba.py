import streamlit as st 
import cv2
from PIL import Image, ImageEnhance
import numpy as np

# Inisialisasi CascadeClassifier
face_cascade = cv2.CascadeClassifier('./detectors/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./detectors/haarcascade_eye.xml')

# Fungsi untuk mendeteksi wajah
def detect_faces(our_image):
    new_img = np.array(our_image.convert("RGB"))
    gray_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY) # Konversi ke skala abu-abu
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor= 1.1, minNeighbors=7)
    for (x, y, w, h) in faces:
        cv2.rectangle(new_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return new_img, faces

# Fungsi untuk mendeteksi mata hanya di dalam area wajah
def detect_eye_within_faces(our_image):
    new_img = np.array(our_image.convert("RGB"))
    gray_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2GRAY) # Konversi ke skala abu-abu
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor= 1.1, minNeighbors=5)

    # Loop melalui setiap wajah yang terdeteksi
    for (x, y, w, h) in faces:
        face_roi = gray_img[y:y+h, x:x+w] # Wilayah wajah dalam gambar abu-abu
        eyes = eye_cascade.detectMultiScale(face_roi, scaleFactor= 1.1, minNeighbors=15)

        # Loop melalui setiap mata yang terdeteksi di dalam wajah
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(new_img, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
    
    return new_img, eyes

# Fungsi utama aplikasi
def main():
    st.title('Image Editing App')
    st.text('Edit your images in a fast and simple way')

    activities = ['Detection', 'About']
    choice = st.sidebar.selectbox('Select Activity', activities)

    if choice == 'Detection':
        st.subheader('Face Detection')
        image_file = st.file_uploader('Upload Image', type=['jpg','png','jpeg'])

        if image_file is not None:
            our_image = Image.open(image_file)
            st.text('Original Image')
            st.image(our_image)

            enhance_type = st.sidebar.radio("Enhance type", ['Original','Gray-scale','Contrast','Brightness','Blurring','Sharpness'])

            if enhance_type == 'Gray-scale':
                img = np.array(our_image.convert('RGB'))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                st.image(gray)
            elif enhance_type == "Contrast":
                rate = st.sidebar.slider("Contrast", 0.5, 3.0)
                enhancer = ImageEnhance.Contrast(our_image)
                enhanced_img = enhancer.enhance(rate)
                st.image(enhanced_img)
            elif enhance_type == "Brightness":
                rate = st.sidebar.slider("Brightness", 0.0, 3.0)
                enhancer = ImageEnhance.Brightness(our_image)
                enhanced_img = enhancer.enhance(rate)
                st.image(enhanced_img)
            elif enhance_type == "Blurring":
                rate = st.sidebar.slider("Blurring", 0.0, 10.0)
                img = np.array(our_image.convert('RGB'))
                blurred_img = cv2.GaussianBlur(img, (11, 11), rate)
                st.image(blurred_img)
            elif enhance_type == "Sharpness":
                rate = st.sidebar.slider("Sharpness", 0.0, 3.0)
                enhancer = ImageEnhance.Sharpness(our_image)
                enhanced_img = enhancer.enhance(rate)
                st.image(enhanced_img)
            elif enhance_type == "Original":
                st.image(our_image)
            else:
                st.image(our_image)

        tasks = ["Faces", "Eyes"]
        feature_choice = st.sidebar.selectbox("Find features", tasks)

        if st.button("Process"):
            if feature_choice == "Faces":
                result_img, result_face = detect_faces(our_image)
                st.image(result_img)
                st.success("{} Wajah Terdeteksi".format(len(result_face)))

            if feature_choice == "Eyes":
                result_img, result_eye = detect_eye_within_faces(our_image)
                st.image(result_img)
                st.success("{} Mata Terdeteksi".format(len(result_eye)))
    
    elif choice == 'About':
        st.subheader('About Developer')
        st.markdown('Built with Streamlit by Gilar Sumilar')
        st.text('My name is Gilar Sumilar and I am Spiderman')

if __name__ == '__main__':
    main()
