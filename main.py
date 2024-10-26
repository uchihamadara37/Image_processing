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




st.title("Aplikasi Streamlit Pertama Saya")
st.write("Halo, dunia!")

# Judul aplikasi
st.title("Upload dan Tampilkan Gambar")


# Display radio button to select mode
mode = st.radio(
    "Choose the mode:",
    ('Original', 'Grayscale', 'Negative', 'Binary', 'Saturation', 'Posterize', 'Histogram Equalization')
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
        saturation = st.slider('Adjust Saturation', min_value=0, max_value=200, value=100)

        img_array = np.array(image.convert('RGB'))
        grayscale = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])
        grayscale = np.stack([grayscale] * 3, axis=-1)

        result = img_array * (saturation / 100) + grayscale * (1 - saturation / 100)
        result = np.clip(result, 0, 255).astype(np.uint8)

        processed_image = Image.fromarray(result)
    elif mode == 'Posterize':
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

    # Create two columns to display original and processed images
    col1, col2 = st.columns(2)
    # Display original image in the first column
    with col1:
        st.image(uploaded_file, caption='Original Image', use_column_width=True)
    # Display processed image in the second column
    with col2:
        st.image(processed_image, caption=f'{mode} Image', use_column_width=True)

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
                    value=100
                )
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
            
    Tambahi dewe ngab

"""
