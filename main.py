import threading

import streamlit as st
import pandas as pd
from PIL import Image, ImageEnhance, ImageOps
import io
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt

def apply_grayscale(img):
    return img.convert('L')

# Atur konfigurasi halaman
st.set_page_config(
    page_title="Citrane dewe",
    layout="wide"  # Pilihan: 'centered' atau 'wide'
)

st.text("made by : ")
st.text("Andrea Alfian Sah Putra    (123220078)")
st.text("Naufal Laudza Rohmadi      (123220093)")
st.text("Sakti Maulana Ibrahim      (123220101)")

def make_histogram(image, judul, st_object):
    # Tampilkan histogram hasil gambar negatif
    fig, ax2 = plt.subplots(figsize=(10, 4))
    colors = ['red', 'green', 'blue']
    for i, color in enumerate(colors):
        hist = cv2.calcHist([np.array(image)], [i], None, [256], [0, 256])
        ax2.plot(hist, color=color, alpha=0.5, label=color.upper())

    ax2.set_title(f'RGB Histogram ({judul})')
    ax2.set_xlabel('Pixel Value')
    ax2.set_ylabel('Frequency')
    ax2.legend()

    st_object.pyplot(fig)

def make_histogram_hitam_putih(image, st_object):
    fig, ax = plt.subplots()
    histogram = image.histogram()

    # Since it's a binary image, we only need the first and last values of the histogram
    black_pixels = histogram[0]
    white_pixels = histogram[-1]

    ax.bar(['Black', 'White'], [black_pixels, white_pixels])
    ax.set_ylabel('Pixel Count')
    ax.set_title('Histogram Hasil Gambar')

    # Display the histogram in Streamlit
    st_object.pyplot(fig)




st.markdown("<h1 style='text-align: center;'>Aplikasi Untuk Mengolah Gambar</h1>", unsafe_allow_html=True)


# Judul aplikasi
st.title("Upload dan Tampilkan Gambar")


# Display radio button to select mode
mode = st.radio(
    "Choose the mode:",
    ('Original', 'Grayscale', 'Negative', 'Binary', 'Saturation', 'Posterize', 'Histogram Equalization','Contrast','Sharpness','Blur','Motion Blur')
)

# File uploader untuk memilih gambar
uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])



# Jika ada file yang diupload
if uploaded_file is not None:
    # Membaca gambar
    # image = Image.open(uploaded_file)
    image = Image.open(io.BytesIO(uploaded_file.getvalue()))



    # Apply selected mode
    processed_image = image
    if mode == 'Grayscale':
        # Create a slider to adjust the grayscale intensity
        grayscale_intensity = st.slider('Adjust Grayscale Intensity', min_value=0, max_value=100, value=100)

        # Convert image ke numpy array
        img_array = np.array(image.convert('RGB'))
        result = img_array.copy()

        # Hitung grayscale menggunakan formula standar
        # Gray = 0.299R + 0.587G + 0.114B
        grayscale = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])

        # Buat array 3D untuk grayscale (sama untuk setiap channel RGB)
        grayscale_rgb = np.stack([grayscale] * 3, axis=-1)

        # Interpolasi antara gambar asli dan grayscale berdasarkan intensitas
        result = img_array * (1 - grayscale_intensity / 100) + grayscale_rgb * (grayscale_intensity / 100)
        result = np.clip(result, 0, 255).astype(np.uint8)

        processed_image = Image.fromarray(result)

    elif mode == 'Negative':
        # Apply negative effect
        negative_intensity = st.slider(
            'Adjust Negative Intensity',
            min_value=0,
            max_value=100,
            value=100
        )

        # Convert image to numpy array
        img_array = np.array(image)
        result = img_array.copy()

        # Terapkan efek negatif per channel
        if negative_intensity > 0:
            negative = 255 - img_array

            # Interpolasi antara gambar asli dan negatif berdasarkan intensitas
            result = img_array * (1 - negative_intensity / 100) + negative * (negative_intensity / 100)
            result = np.clip(result, 0, 255).astype(np.uint8)

        # Konversi kembali ke PIL Image
        processed_image = Image.fromarray(result)



    elif mode == 'Binary':
        # hitam putih
        # Slider untuk intensitas dan threshold
        # binary_intensity = st.slider('Adjust Binary Intensity', min_value=0, max_value=100, value=100)
        binary_intensity = 100
        threshold = st.slider('Threshold', min_value=0, max_value=255, value=128)

        # Convert image ke numpy array
        img_array = np.array(image.convert('RGB'))
        result = img_array.copy()

        # Hitung grayscale dulu (karena biner biasanya dari grayscale)
        grayscale = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])

        # Terapkan threshold untuk mendapatkan gambar biner (hitam putih)
        binary = np.where(grayscale > threshold, 255, 0)

        # Buat array 3D untuk binary (sama untuk setiap channel RGB)
        binary_rgb = np.stack([binary] * 3, axis=-1)

        # Interpolasi antara gambar asli dan binary berdasarkan intensitas
        result = img_array * (1 - binary_intensity / 100) + binary_rgb * (binary_intensity / 100)
        result = np.clip(result, 0, 255).astype(np.uint8)

        processed_image = Image.fromarray(result)
    elif mode == 'Saturation':
        # itensitas atau kekuatan warna
        saturation = st.slider('Adjust Saturation', min_value=0, max_value=200, value=100)
        # misal panjang x lebar x 4 (rgba)
        img_array = np.array(image.convert('RGB'))
        grayscale = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])
        grayscale = np.stack([grayscale] * 3, axis=-1)

        result = img_array * (saturation / 100) + grayscale * (1 - saturation / 100)
        result = np.clip(result, 0, 255).astype(np.uint8)

        processed_image = Image.fromarray(result)

    elif mode == 'Posterize':
        # mengurangi jumlah warna
        levels = st.slider('Color Levels', min_value=2, max_value=8, value=4)
        img_array = np.array(image.convert('RGB'))
        result = (img_array // (256 // levels)) * (256 // levels)
        result = result.astype(np.uint8)

        processed_image = Image.fromarray(result)

    elif mode == 'Histogram Equalization':
        img_array = np.array(image.convert('RGB'))
        # Fungsi untuk equalization pada satu channel
        def equalize_hist(channel):
            # Hitung histogram
            hist, bins = np.histogram(channel.flatten(), bins=256, range=(0, 256))

            # Hitung cumulative distribution function (cdf)
            cdf = hist.cumsum()

            # Normalisasi cdf ke range 0-255
            cdf_normalized = ((cdf - cdf.min()) * 255) / (cdf.max() - cdf.min())

            # Mapping nilai pixel menggunakan cdf
            equalized_channel = cdf_normalized[channel]

            return equalized_channel.astype(np.uint8)


        # Equalize setiap channel RGB
        result = np.zeros_like(img_array)
        for i in range(3):  # untuk setiap channel RGB
            result[:, :, i] = equalize_hist(img_array[:, :, i])

        # Buat slider untuk blend dengan gambar asli
        intensity = st.slider('Equalization Intensity', min_value=0, max_value=100, value=100)

        # Blend hasil equalization dengan gambar asli
        result = img_array * (1 - intensity / 100) + result * (intensity / 100)
        result = np.clip(result, 0, 255).astype(np.uint8)

        processed_image = Image.fromarray(result)

    elif mode == 'Contrast':
        # Tambahkan slider untuk mengatur kontras
        contrast_level = st.slider('Adjust Contrast', min_value=0.0, max_value=2.0, value=1.0)
        enhancer = ImageEnhance.Contrast(image)

        processed_image = enhancer.enhance(contrast_level)

    elif mode == 'Sharpness':
        # Slider untuk mengatur ketajaman
        sharpness_level = st.slider('Adjust Sharpness', min_value=1.0, max_value=5.0, value=1.0)

        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)

        processed_image = enhancer.enhance(sharpness_level)

    elif mode == 'Blur':
        # Slider untuk mengatur intensitas blur
        blur_intensity = st.slider('Adjust Blur Intensity', min_value=1, max_value=20, value=1, step=1)

        # Convert image ke numpy array
        img_array = np.array(image.convert('RGB'))

        # Terapkan Gaussian Blur pada gambar
        blurred_image = cv2.GaussianBlur(img_array, (blur_intensity * 2 + 1, blur_intensity * 2 + 1), 0)

        # Konversi kembali ke PIL Image
        processed_image = Image.fromarray(blurred_image)

    if mode == 'Motion Blur':
        # Slider untuk mengatur intensitas dan arah motion blur
        motion_intensity = st.slider('Adjust Motion Blur Intensity', min_value=1, max_value=20, value=5, step=1)
        direction = st.radio("Choose direction:", ('Horizontal', 'Vertical'))

        # Konversi gambar ke array numpy
        img_array = np.array(image.convert('RGB'))

        # Buat kernel untuk efek motion blur
        if direction == 'Horizontal':
            kernel_motion_blur = np.zeros((1, motion_intensity))
            kernel_motion_blur[0, :] = 1.0 / motion_intensity
        else:  # Vertical direction
            kernel_motion_blur = np.zeros((motion_intensity, 1))
            kernel_motion_blur[:, 0] = 1.0 / motion_intensity

        # Terapkan filter motion blur pada gambar
        blurred_image = cv2.filter2D(img_array, -1, kernel_motion_blur)

        # Konversi kembali ke format PIL Image
        processed_image = Image.fromarray(blurred_image)

    # Create two columns to display original and processed images
    col1, col2 = st.columns(2)
    # Display original image in the first column
    with col1:
        st.image(uploaded_file, caption='Original Image', use_column_width=True)
    # Display processed image in the second column
    with col2:
        st.image(processed_image, caption=f'{mode} Image', use_column_width=True)

        #download
        # Konversi processed_image menjadi buffer untuk pengunduhan
        image_buffer = io.BytesIO()
        processed_image.save(image_buffer, format="PNG")
        image_buffer.seek(0)

        # Tombol untuk mengunduh gambar hasil pemrosesan
        st.download_button(
            label="Download Hasil Gambar",
            data=image_buffer,
            file_name=f"processed_image_{mode}.png",
            mime="image/png"
        )
        #akhir download

    col_histo1, col_histo2 = st.columns(2)
    # menampilkan histogram
    with col_histo1:
        make_histogram(image, "Gambar Asli", st)
    with col_histo2:
        if mode == 'Binary':
            make_histogram_hitam_putih(processed_image, st)
        else:
            make_histogram(processed_image, "Setelah Proses", st)



# =====================================================================

st.title("Gunakan Kamera Real Time ^^")

if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

camera_active = st.checkbox('Activate Camera')

stop_button_placeholder = st.empty()

cap = cv2.VideoCapture(0)
if camera_active:
    st.session_state.camera_active = True

    if not cap.isOpened():
        st.error("Tidak dapat membuka kamera. Pastikan kamera terhubung dan tidak digunakan oleh aplikasi lain.")
    else:
        st.success("Kamera aktif. Uncheck 'Aktifkan Kamera' untuk menghentikan.")

        try:
            if mode == 'Grayscale':
                grayscale_intensity = st.slider(
                    'Adjust Grayscale Intensity',
                    min_value=0,
                    max_value=100,
                    value=100
                )
                col_vid_gray1, col_vid_gray2 = st.columns(2)
                with col_vid_gray1:
                    frame_placeholder = st.empty()
                    frame_hist = st.empty()
                with col_vid_gray2:
                    frame_placeholder2 = st.empty()
                    frame_hist2 = st.empty()

            elif mode == 'Negative':
                # Implementasi mode negative di sini
                negative_intensity = st.slider(
                    'Adjust Negative Intensity',
                    min_value=0,
                    max_value=100,
                    value=100)
                col_vid_gray1, col_vid_gray2 = st.columns(2)
                with col_vid_gray1:
                    frame_placeholder = st.empty()
                    frame_hist = st.empty()
                with col_vid_gray2:
                    frame_placeholder2 = st.empty()
                    frame_hist2 = st.empty()
            elif mode == 'Binary':
                # Implementasi mode binary di sini
                threshold = st.slider('Threshold', min_value=0, max_value=255, value=128)
                col_vid_gray1, col_vid_gray2 = st.columns(2)
                with col_vid_gray1:
                    frame_placeholder = st.empty()
                    frame_hist = st.empty()
                with col_vid_gray2:
                    frame_placeholder2 = st.empty()
                    frame_hist2 = st.empty()

            elif mode == 'Saturation':
                saturation = st.slider('Adjust Saturation', min_value=0, max_value=200, value=100)

                # Create columns for original and processed frame
                col_vid_sat1, col_vid_sat2 = st.columns(2)
                with col_vid_sat1:
                    frame_placeholder = st.empty()
                    frame_hist = st.empty()
                with col_vid_sat2:
                    frame_placeholder2 = st.empty()
                    frame_hist2 = st.empty()

            elif mode == 'Posterize':
                levels = st.slider('Color Levels', min_value=2, max_value=8, value=4)
                col_vid_post1, col_vid_post2 = st.columns(2)
                with col_vid_post1:
                    frame_placeholder = st.empty()
                    frame_hist = st.empty()
                with col_vid_post2:
                    frame_placeholder2 = st.empty()
                    frame_hist2 = st.empty()

            elif mode == 'Histogram Equalization':
                # Slider untuk intensitas equalization
                intensity = st.slider('Equalization Intensity', min_value=0, max_value=100, value=100)

                # Buat dua kolom untuk frame asli dan hasil histogram equalization
                col_vid_eq1, col_vid_eq2 = st.columns(2)
                with col_vid_eq1:
                    frame_placeholder = st.empty()
                    frame_hist = st.empty()
                with col_vid_eq2:
                    frame_placeholder2 = st.empty()
                    frame_hist2 = st.empty()

            elif mode == 'Contrast':
                contrast_level = st.slider('Adjust Contrast', min_value=0.5, max_value=2.0, value=1.0)

                # Membuat dua kolom untuk frame asli dan hasil kontras
                col_vid_contrast1, col_vid_contrast2 = st.columns(2)
                with col_vid_contrast1:
                    frame_placeholder = st.empty()
                    frame_hist = st.empty()
                with col_vid_contrast2:
                    frame_placeholder2 = st.empty()
                    frame_hist2 = st.empty()

            elif mode == 'Sharpness':
                sharpness_level = st.slider('Adjust Sharpness', min_value=0.0, max_value=5.0, value=1.0)

                # Membuat dua kolom untuk frame asli dan hasil kontras
                col_vid_contrast1, col_vid_contrast2 = st.columns(2)
                with col_vid_contrast1:
                    frame_placeholder = st.empty()
                    frame_hist = st.empty()
                with col_vid_contrast2:
                    frame_placeholder2 = st.empty()
                    frame_hist2 = st.empty()

            elif mode == 'Blur':
                # Slider untuk mengatur intensitas blur
                blur_intensity = st.slider('Adjust Blur Intensity', min_value=1, max_value=20, value=1, step=1)

                # Membuat dua kolom untuk frame asli dan hasil kontras
                col_vid_contrast1, col_vid_contrast2 = st.columns(2)
                with col_vid_contrast1:
                    frame_placeholder = st.empty()
                    frame_hist = st.empty()
                with col_vid_contrast2:
                    frame_placeholder2 = st.empty()
                    frame_hist2 = st.empty()

            elif mode == 'Motion Blur':
                # Slider untuk mengatur intensitas dan arah motion blur
                motion_intensity = st.slider('Adjust Motion Blur Intensity', min_value=1, max_value=20, value=5, step=1)
                direction = st.radio("Choose direction:", ('Horizontal', 'Vertical'))

                # Membuat dua kolom untuk frame asli dan hasil motion blur
                col_vid_motion1, col_vid_motion2 = st.columns(2)
                with col_vid_motion1:
                    frame_placeholder = st.empty()
                    frame_hist = st.empty()
                with col_vid_motion2:
                    frame_placeholder2 = st.empty()
                    frame_hist2 = st.empty()

            else:
                # papan
                frame_placeholder = st.empty()
                frame_placeholder2 = st.empty()

            while st.session_state.camera_active:
                ret, frame = cap.read()
                if ret:
                    # Konversi BGR ke RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.flip(frame, 1)
                    # Tampilkan frame
                    frame_placeholder.image(frame, channels="RGB", use_column_width=True)
                    if mode == 'Grayscale':
                        # Convert frame to numpy array
                        img_array = np.array(frame)

                        # Hitung grayscale menggunakan formula standar
                        # Gray = 0.299R + 0.587G + 0.114B
                        grayscale = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])

                        # Buat array 3D untuk grayscale (sama untuk setiap channel RGB)
                        grayscale_rgb = np.stack([grayscale] * 3, axis=-1)

                        # Interpolasi antara gambar asli dan grayscale berdasarkan intensitas
                        result = img_array * (1 - grayscale_intensity / 100) + grayscale_rgb * (
                                    grayscale_intensity / 100)
                        result = np.clip(result, 0, 255).astype(np.uint8)

                        # Tampilkan frame grayscale
                        frame_placeholder2.image(result, channels="RGB", use_column_width=True)
                        make_histogram(frame, "Histogram Video Asli", frame_hist)
                        make_histogram(result, "Histogram Video Hasil", frame_hist2)


                    elif mode == 'Negative':
                        # Implementasi mode negative di sini
                        # Convert image to numpy array
                        img_array = np.array(frame)
                        result = img_array.copy()

                        # Terapkan efek negatif per channel
                        negative = 255 - img_array

                        # Interpolasi antara gambar asli dan negatif berdasarkan intensitas
                        result = img_array * (1 - negative_intensity / 100) + negative * (negative_intensity / 100)
                        result = np.clip(result, 0, 255).astype(np.uint8)

                        # Konversi kembali ke PIL Image
                        # processed_image = Image.fromarray(result)

                        frame_placeholder2.image(result, channels="RGB", use_column_width=True)
                        make_histogram(frame, "Histogram Video Asli", frame_hist)
                        make_histogram(result, "Histogram Video Hasil", frame_hist2)

                    elif mode == 'Binary':
                        # Implementasi mode binary di sini
                        binary_intensity = 100

                        # Convert image ke numpy array
                        img_array = np.array(frame)
                        result = img_array.copy()

                        # Hitung grayscale dulu (karena biner biasanya dari grayscale)
                        grayscale = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])

                        # Terapkan threshold untuk mendapatkan gambar biner (hitam putih)
                        binary = np.where(grayscale > threshold, 255, 0)

                        # Buat array 3D untuk binary (sama untuk setiap channel RGB)
                        binary_rgb = np.stack([binary] * 3, axis=-1)

                        # Interpolasi antara gambar asli dan binary berdasarkan intensitas
                        result = img_array * (1 - binary_intensity / 100) + binary_rgb * (binary_intensity / 100)
                        result = np.clip(result, 0, 255).astype(np.uint8)
                        processed_image = Image.fromarray(result)

                        frame_placeholder2.image(result, channels="RGB", use_column_width=True)
                        make_histogram(frame, "Histogram Video Asli", frame_hist)
                        make_histogram_hitam_putih(processed_image, frame_hist2)

                    elif mode == 'Saturation':
                        # Convert frame to numpy array
                        img_array = np.array(frame)
                        grayscale = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])
                        grayscale = np.stack([grayscale] * 3, axis=-1)

                        # Apply saturation adjustment
                        result = img_array * (saturation / 100) + grayscale * (1 - saturation / 100)
                        result = np.clip(result, 0, 255).astype(np.uint8)

                        # Display original and processed frames
                        frame_placeholder.image(frame, channels="RGB", use_column_width=True)
                        frame_placeholder2.image(result, channels="RGB", use_column_width=True)

                        # Display histograms
                        make_histogram(frame, "Histogram Video Asli", frame_hist)
                        make_histogram(result, "Histogram Video Hasil", frame_hist2)

                    elif mode == 'Posterize':
                        # Konversi frame ke array numpy
                        img_array = np.array(frame)

                        # Terapkan efek posterize dengan membagi nilai pixel menjadi beberapa level
                        result = (img_array // (256 // levels)) * (256 // levels)
                        result = result.astype(np.uint8)

                        # Tampilkan frame asli dan hasil posterize
                        frame_placeholder.image(frame, channels="RGB", use_column_width=True)
                        frame_placeholder2.image(result, channels="RGB", use_column_width=True)

                        # Tampilkan histogram untuk frame asli dan hasil posterize
                        make_histogram(frame, "Histogram Video Asli", frame_hist)
                        make_histogram(result, "Histogram Video Hasil", frame_hist2)

                    elif mode == 'Histogram Equalization':
                        # Konversi frame ke array numpy
                        img_array = np.array(frame)


                        # Fungsi untuk melakukan equalization pada satu channel
                        def equalize_hist(channel):
                            # Hitung histogram
                            hist, bins = np.histogram(channel.flatten(), bins=256, range=(0, 256))
                            # Hitung cumulative distribution function (cdf)
                            cdf = hist.cumsum()
                            # Normalisasi cdf ke range 0-255
                            cdf_normalized = ((cdf - cdf.min()) * 255) / (cdf.max() - cdf.min())
                            # Mapping nilai pixel menggunakan cdf
                            return cdf_normalized[channel].astype(np.uint8)


                        # Equalize setiap channel RGB
                        result = np.zeros_like(img_array)
                        for i in range(3):  # untuk setiap channel RGB
                            result[:, :, i] = equalize_hist(img_array[:, :, i])

                        # Blend hasil equalization dengan frame asli sesuai dengan intensitas
                        result = img_array * (1 - intensity / 100) + result * (intensity / 100)
                        result = np.clip(result, 0, 255).astype(np.uint8)

                        # Tampilkan frame asli dan hasil equalization
                        frame_placeholder.image(frame, channels="RGB", use_column_width=True)
                        frame_placeholder2.image(result, channels="RGB", use_column_width=True)

                        # Tampilkan histogram untuk frame asli dan hasil equalization
                        make_histogram(frame, "Histogram Video Asli", frame_hist)
                        make_histogram(result, "Histogram Video Hasil", frame_hist2)

                    elif mode == 'Contrast':
                        img_pil = Image.fromarray(frame)

                        # Terapkan pengaturan kontras menggunakan ImageEnhance
                        enhancer = ImageEnhance.Contrast(img_pil)
                        contrast_img = enhancer.enhance(contrast_level)

                        # Konversi kembali ke format numpy untuk ditampilkan di Streamlit
                        result = np.array(contrast_img)

                        # Tampilkan gambar hasil dengan kontras yang disesuaikan
                        frame_placeholder2.image(result, channels="RGB", use_column_width=True)

                        # Tampilkan histogram asli dan hasil kontras
                        make_histogram(frame, "Histogram Video Asli", frame_hist)
                        make_histogram(result, "Histogram Video Hasil", frame_hist2)

                    elif mode == 'Sharpness':
                        img_pil = Image.fromarray(frame)

                        # Aplikasikan Sharpeness
                        enhancer = ImageEnhance.Sharpness(img_pil)
                        result = enhancer.enhance(sharpness_level)

                        # Konversi kembali ke nump array
                        result = np.array(result)

                        # Tampilkan Hasil
                        frame_placeholder2.image(result, channels="RGB", use_column_width=True)

                        # Tampilkan original frame
                        frame_placeholder.image(frame, channels="RGB", use_column_width=True)

                        # Tampilkan histogram asli dan hasil kontras
                        make_histogram(frame, "Histogram Video Asli", frame_hist)
                        make_histogram(result, "Histogram Video Hasil", frame_hist2)

                    elif mode == 'Blur':
                        # Konversi frame ke array numpy
                        img_array = np.array(frame)

                        # Terapkan Gaussian Blur pada gambar
                        blurred_image = cv2.GaussianBlur(img_array, (blur_intensity * 2 + 1, blur_intensity * 2 + 1), 0)

                        # Konversi kembali ke PIL Image
                        processed_image = Image.fromarray(blurred_image)

                        # Tampilkan Hasil
                        frame_placeholder2.image(processed_image, channels="RGB", use_column_width=True)

                        # Tampilkan original frame
                        frame_placeholder.image(frame, channels="RGB", use_column_width=True)

                        # Tampilkan histogram asli dan hasil kontras
                        make_histogram(frame, "Histogram Video Asli", frame_hist)
                        make_histogram(processed_image, "Histogram Video Hasil", frame_hist2)

                    if mode == 'Motion Blur':
                        # Buat kernel untuk efek motion blur berdasarkan arah
                        if direction == 'Horizontal':
                            kernel_motion_blur = np.zeros((1, motion_intensity))
                            kernel_motion_blur[0, :] = 1.0 / motion_intensity
                        else:  # Vertical
                            kernel_motion_blur = np.zeros((motion_intensity, 1))
                            kernel_motion_blur[:, 0] = 1.0 / motion_intensity

                        # Terapkan filter motion blur pada frame
                        blurred_frame = cv2.filter2D(frame, -1, kernel_motion_blur)

                        # Tampilkan original frame
                        frame_placeholder.image(frame, channels="RGB", use_column_width=True)

                        # Tampilkan hasil frame dengan motion blur
                        frame_placeholder2.image(blurred_frame, channels="RGB", use_column_width=True)

                        # Tampilkan histogram asli dan hasil motion blur
                        make_histogram(frame, "Histogram Video Asli", frame_hist)
                        make_histogram(blurred_frame, "Histogram Video Hasil", frame_hist2)

                    else:
                        frame_placeholder.image(frame, channels="RGB", use_column_width=True)
                        # Mode Original, tidak perlu pemrosesan tambahan
                        frame_placeholder2.empty()

                    # Tombol stop
                else:
                    st.error("Tidak dapat membaca frame dari kamera.")
                    break

            print("berhasil dimatikan")
            cap.release()
        except Exception as e:
            st.error(f"Error: {str(e)}")
            cap.release()

else:
    cap.release()
    st.session_state.camera_active = False
    st.write("Kamera tidak aktif.")



"""
    Catatan ***
    
        Histogram equalization adalah teknik pemrosesan citra yang digunakan 
        untuk meningkatkan kontras gambar. 
        Dalam pendekatan ini, teknik histogram equalization mengatur distribusi 
        intensitas piksel sehingga lebih merata. Ini sering diterapkan pada citra 
        grayscale tetapi dapat diterapkan pula pada citra berwarna (RGB). 
        Berikut penjelasan konsep dasar histogram equalization pada gambar 
        dengan intensitas warna RGB 0-255:

        Konversi ke Grayscale (Opsional): 
            Histogram equalization lebih sederhana jika diterapkan pada citra grayscale. 
            Citra berwarna biasanya dikonversi ke grayscale terlebih dahulu, 
            meskipun ada metode untuk menerapkannya langsung pada citra berwarna (RGB).

        Histogram dan Cumulative Distribution Function (CDF):
            Histogram menunjukkan jumlah piksel pada setiap level intensitas 
            (0 hingga 255 untuk RGB).
            CDF dihitung berdasarkan histogram ini untuk merepresentasikan 
            distribusi kumulatif dari intensitas.
        
        Transformasi Histogram:
            CDF dinormalisasi untuk menghasilkan intensitas baru untuk setiap piksel. 
            Proses ini membantu mendistribusikan intensitas lebih merata, 
            dengan cara membuat piksel yang gelap menjadi lebih terang dan sebaliknya.

        Penerapan pada RGB:
                Jika dilakukan langsung pada komponen warna RGB, 
            histogram equalization diterapkan pada setiap kanal warna (R, G, dan B) secara terpisah. 
            Namun, ini bisa menghasilkan warna yang kurang alami.
            Alternatifnya, konversi citra ke ruang warna yang memisahkan intensitas 
            dan warna (misalnya YCbCr atau HSV), dan menerapkan equalization hanya 
            pada kanal intensitas (Y atau V) agar warna tidak terdistorsi.
                Dengan histogram equalization, gambar yang semula tampak terlalu gelap 
            atau terlalu terang akan terlihat lebih jelas, dengan distribusi intensitas 
            yang lebih seimbang.
            
        Gaussian Blur
            Gaussian Blur adalah teknik pengaburan gambar yang menggunakan distribusi Gaussian
            (atau distribusi normal) untuk menghaluskan gambar. 
            Dalam teknik ini setiap piksel gambar dihitung ulang dengan mengambil nilai rata-rata
            dari piksel-piksel tetangganya, dimana tetangga yang lebih dekat memiliki bobot yang
            lebih besar sesuai dengan kurva Gaussian. Ini mengakibatkan efek pengaburan yang 
            lembut dan alami.
            
            Di sini, fungsi cv2.GaussianBlur dari OpenCV diterapkan pada gambar. Kernel atau ukuran
            filter blur yang diterapkan ditentukan oleh (blur_intensity * 2 + 1, blur_intensity * 2 + 1),
            di mana nilai ini selalu ganjil (syarat kernel Gaussian di OpenCV). Semakin besar nilai
            blur_intensity, semakin kabur gambarnya.
            
            
        Motion Blur
            Kernel berfungsi untuk "membaurkan" atau "membaurkan" piksel gambar
            di sekitar piksel target dengan cara tertentu untuk menciptakan efek blur. 
            Pada motion blur, kita mengaburkan gambar seolah-olah objek dalam gambar
            bergerak dengan cepat dalam satu arah tertentu.
            
            np.zeros((1, motion_intensity)): Membuat sebuah matriks kosong 
            dengan 1 baris dan sebanyak motion_intensity kolom. 
            Matriks ini adalah kernel yang akan diterapkan pada gambar.

            Misalnya, jika motion_intensity = 5, 
            maka ukuran kernel akan menjadi (1, 5), yaitu 1 baris dan 5 kolom.
            kernel_motion_blur[0, :] = 1.0 / motion_intensity:

            Mengisi semua elemen dalam baris pertama 
            (satu-satunya baris dalam kasus horizontal)
            dengan nilai 1.0 / motion_intensity.
            Ini berarti setiap elemen dalam baris kernel akan memiliki bobot yang sama, 
            dan total bobotnya adalah 1 (karena 1.0 / 5 + 1.0 / 5 + ... + 1.0 / 5 = 1).
"""
