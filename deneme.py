import tkinter as tk  # tkinter kütüphanesini içe aktarır (GUI oluşturmak için)
from tkinter import filedialog, messagebox  # dosya seçme ve mesaj kutuları için ek özellikler

import customtkinter as ctk  # özel tkinter işlevleri

import cv2  # Görüntü işleme için OpenCV kütüphanesi
import numpy as np  # Bilimsel hesaplamalar için numpy kütüphanesi
from PIL import Image, ImageFilter, ImageTk  # Resim işleme ve gösterme için PIL kütüphanesi

from matplotlib import pyplot as plt


class ImageFilterApp:
    def __init__(self, root):

        self.root = root  # Ana pencereyi (root) saklar
        self.root.title("Resim Filtreleme ve Karşılaştırma")  # Pencere başlığını ayarlar

        self.file_path = None  # Dosya yolu bilgisini saklar
        self.original_img = None  # Orijinal görüntüyü saklar
        self.filtered_img = None  # Filtrelenmiş görüntüyü saklar
        self.camera = None  # Kamera görüntüsünü saklar (eğer kullanılıyorsa)

        self.init_ui()  # Kullanıcı arayüzünü başlatmak için metodu çağırır


    def init_ui(self):
        # Ana butonları içeren bir frame oluşturuluyor ve yerleştiriliyor
        button_frame = tk.Frame(self.root)
        button_frame.pack(side=tk.LEFT, padx=10, pady=10) #oluşturulan frame'in konumlandırması ve boyutlandırılması yapılıyor.

        # Kamera açmak için buton oluşturuluyor.
        self.camera_button = ctk.CTkButton(button_frame, text="Kamera", command=self.open_camera)
        self.camera_button.pack(padx=5, pady=5)

        # Resim seçmek için buton oluşturuluyor
        self.file_button = ctk.CTkButton(button_frame, text='Resim Seç', command=self.open_file_dialog)
        self.file_button.pack(padx=5, pady=5)

        # Filtreleme işlemlerinin seçilebileceği liste oluşturuluyor.
        self.filter_choice = tk.StringVar()
        self.filter_choice.set("")  # Başlangıçta varsayılan olarak Blur seçili
        filter_options = ["Blur", "Sharpen", "Contour", "Adaptive Threshold", "Add-Border", "Gamma Fix", "Sobel Edge Detection", "Laplacian Edge Detection", "Canny Edge Detection", "Deriche Edge Detection", "Harris Corner Detection", "Cascade Classifier", "Contour Detection", "Watershed"]
        self.filter_dropdown = tk.OptionMenu(button_frame, self.filter_choice, *filter_options, command=self.show_params)
        self.filter_dropdown.pack(padx=5, pady=5)

        # Block Size parametresi için giriş oluşturuluyor.
        self.param_label_1 = ctk.CTkLabel(button_frame, text="Block Size:")
        self.block_size_entry = ctk.CTkEntry(button_frame)

        # C parametresi için buton oluşturuluyor
        self.param_label_2 = ctk.CTkLabel(button_frame, text="C:")
        self.c_entry = ctk.CTkEntry(button_frame)

        # Kenar rengi için giriş oluşturuluyor ve içine nasıl girilmesi gerektiğiyle alakalı bir örnek konuyor. (B,G,R)
        self.border_color_label = ctk.CTkLabel(button_frame, text="Border Color:")
        self.border_color_entry = ctk.CTkEntry(button_frame)
        self.border_color_entry.insert(0, '255,0,0')

        # Kenar kalınlığı parametresi için giriş oluşturuluyor.
        self.border_width_label = ctk.CTkLabel(button_frame, text="Border Width:")
        self.border_width_entry = ctk.CTkEntry(button_frame)

        # Bulanıklaştırma filtresinin kendi içindeki seçenecekleri için bir seçim listesi oluşturuluyor.
        self.filter_choice_blur_label = ctk.CTkLabel(button_frame, text="Choise Blur Type:")
        self.filter_choice_blur = tk.StringVar()
        self.filter_choice_blur.set("Blur")  # Başlangıçta varsayılan olarak Blur seçiliyor.
        blur_options = ["Blur", "Median Blur", "Box Filter", "bilateralFilter", "GaussianBlur"]
        self.blur_filter_dropdown = tk.OptionMenu(button_frame, self.filter_choice_blur, *blur_options, command=self.show_params)
        self.blur_filter_dropdown.pack(padx=5, pady=5)

        # Keskinleştirme filtresinin kendi içindeki seçenekleri için bir seçim listesi oluşturuluyor.
        self.filter_choice_sharpen_label = ctk.CTkLabel(button_frame, text="Choise Sharpen Type:")
        self.filter_choice_sharpen = tk.StringVar()
        self.filter_choice_sharpen.set("Sharpen")  # Başlangıçta varsayılan olarak Sharpen seçiliyor.
        sharpen_options = ["Sharpen", "Outline"]
        self.sharpen_filter_dropdown = tk.OptionMenu(button_frame, self.filter_choice_sharpen, *sharpen_options, command=self.show_params)
        self.sharpen_filter_dropdown.pack(padx=5, pady=5)

        # Filtrelemenin farklı çözümlerinin ekrana yansıtılabilmesi için liste oluşturuluyor.
        self.filter_choice_solution_label = ctk.CTkLabel(button_frame, text="Choise Solution")
        self.filter_choice_solution = tk.StringVar()
        self.filter_choice_solution.set("Solution-1")  # Başlangıçta varsayılan olarak 1. çözüm seçiliyor.
        solution_options = ["Solution-1", "Solution-2"]
        self.solution_filter_dropdown = tk.OptionMenu(button_frame, self.filter_choice_solution, *solution_options, command=self.show_params)
        self.solution_filter_dropdown.pack(padx=5, pady=5)

        # Conf parametresinin çözümleri için liste oluşturuluyor.
        self.conf_choice_label = ctk.CTkLabel(button_frame, text="Choise Conf")
        self.conf_choice_solution = tk.StringVar()
        self.conf_choice_solution.set("Conf-1")  # Başlangıçta varsayılan olarak Conf-1 seçiliyor.
        conf_options = ["Conf-1", "Conf-2"]
        self.conf_choice_dropdown = tk.OptionMenu(button_frame, self.conf_choice_solution, *conf_options, command=self.show_params)
        self.conf_choice_dropdown.pack(padx=5, pady=5)

        # Kernel boyutu parametresi için giriş oluşturuluyor.
        self.kernel_size_label = ctk.CTkLabel(button_frame, text="Kernel Size:")
        self.kernel_size_entry = ctk.CTkEntry(button_frame)

        # Gamma değeri için giriş oluşturuluyor.
        self.gamma_value_label = ctk.CTkLabel(button_frame, text="Gamma Value:")
        self.gamma_value_entry = ctk.CTkEntry(button_frame)

        # Alçak ve Yüksek Threshold değerleri için girişler oluşturuluyor.
        self.low_threshold_label = ctk.CTkLabel(button_frame, text="Low Threshold Value:")
        self.low_threshold_entry = ctk.CTkEntry(button_frame)
        self.high_threshold_label = ctk.CTkLabel(button_frame, text="High Threshold Value:")
        self.high_threshold_entry = ctk.CTkEntry(button_frame)

        # Alpha değeri için giriş oluşturuluyor.
        self.alpha_label = ctk.CTkLabel(button_frame, text="Alpha Value:")
        self.alpha_entry = ctk.CTkEntry(button_frame)

        # Köşe Kalitesi-Minimum Mesafe-Blok Boyutu parametreleri için girişler oluşturuluyor ve içine örnek girdiler konuluyor.
        self.corner_quality_label = ctk.CTkLabel(button_frame, text="Corner Quality Value:")
        self.corner_quality_entry = ctk.CTkEntry(button_frame)
        self.corner_quality_entry.insert(0, '0.04')
        self.min_distance_label = ctk.CTkLabel(button_frame, text="Min Distance Value:")
        self.min_distance_entry = ctk.CTkEntry(button_frame)
        self.min_distance_entry.insert(0, '10')
        self.block_size_label = ctk.CTkLabel(button_frame, text="Block Size Value:")
        self.block_size_entry = ctk.CTkEntry(button_frame)
        self.block_size_entry.insert(0, '3')

        # Seçilen filtreyi uygulamak için buton oluşturuluyor.
        self.apply_filter_button = ctk.CTkButton(button_frame, text='Filtre Uygula', command=self.apply_filter)
        self.apply_filter_button.pack(padx=5, pady=5)

        # Uygulanan filtreyi kaydetmek için buton oluşturuluyor.
        self.save_button = ctk.CTkButton(button_frame, text='Kaydet', command=self.save_image)
        self.save_button.pack(padx=5, pady=5)

        # Orijinal ve filtrelenmiş resimleri gösterecek label'lar oluşturuluyor ve paketleniyor
        self.original_label = tk.Label(self.root)
        self.original_label.pack(side=tk.LEFT, padx=10)
        self.filtered_label = tk.Label(self.root)
        self.filtered_label.pack(side=tk.LEFT, padx=10)

        self.hide_params()

    def open_camera(self):
        self.camera = cv2.VideoCapture(0)  # 0 numarasıyla kamera kaynağını açar (genellikle yerleşik kamerayı temsil eder)
        self.continuous_camera_capture()  # Ardışık olarak kamera görüntüsünü yakalamaya başlar

    def take_photo_from_camera(self):
        return_value, frame = self.camera.read()  # Kameradan bir kare okur

        if return_value:  # Okuma başarılıysa devam eder
            cv2.imwrite("captured_image.png", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # Kareyi diske kaydeder
            self.original_img = Image.open("captured_image.png")  # Kaydedilen kareyi açar
            resized_img = self.original_img.resize((400, 400), Image.LANCZOS)  # Boyutlandırma işlemi yapar
            photo = ImageTk.PhotoImage(resized_img)  # Tkinter için uygun formata dönüştürür
            self.original_label.configure(image=photo)  # Etiketi (label) yeniden ayarlar ve görüntüyü gösterir
            self.original_label.image = photo  # Etiketi (label) günceller
        else:
            print("Kamera görüntüsü alınamadı.")  # Eğer görüntü alınamazsa hata mesajı verir

    def continuous_camera_capture(self):
        messagebox.showinfo("Uyarı",
                            "Fotoğraf çekmek için 'c' tuşuna basınız.")  # 'c' tuşuna basınca fotoğraf çekme uyarısı verir
        while True:
            return_value, frame = self.camera.read()  # Kameradan bir kare okur
            cv2.imshow("Kamera", frame)  # Kameradan gelen görüntüyü gösterir

            key = cv2.waitKey(1)  # Kullanıcının tuş girişlerini kontrol eder
            if key == ord('c'):  # Eğer 'c' tuşuna basılırsa
                cv2.destroyAllWindows()  # Tüm pencereleri kapatır
                self.take_photo_from_camera()  # Kameradan fotoğraf çeker ve gösterir
                break  # Döngüyü sonlandırır

        self.camera.release()  # Kamera kaynağını serbest bırakır

    @staticmethod
    def pil_to_opencv(pil_image):
        # PIL formatındaki bir resmi NumPy dizisine dönüştürür ve renk formatını değiştirerek OpenCV için uygun hale getirir
        open_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return open_cv_image  # Dönüştürülen resmi döndürür

    # Burada yapılan işlem bazı filtreler için gereken parametreleri o filtre seçili değilse ekranda göstermesin.
    # Bu işlem için gereken tüm kodları hide_params fonksiyonu içinde yaptık.
    def hide_params(self):
        self.param_label_1.pack_forget()
        self.param_label_2.pack_forget()
        self.block_size_entry.pack_forget()
        self.c_entry.pack_forget()

        if self.filter_choice.get() != "Blur":  # Eğer Blur seçili değilse dropdown listeyi gizle
            self.blur_filter_dropdown.pack_forget()
            self.filter_choice_blur_label.pack_forget()
            self.kernel_size_label.pack_forget()
            self.kernel_size_entry.pack_forget()
        if self.filter_choice.get() != "Sharpen":
            self.sharpen_filter_dropdown.pack_forget()
            self.filter_choice_sharpen_label.pack_forget()
        if self.filter_choice.get() != "Gamma Fix":
            self.gamma_value_label.pack_forget()
            self.gamma_value_entry.pack_forget()
        if self.filter_choice.get() != "Add-Border":
            self.border_color_label.pack_forget()
            self.border_color_entry.pack_forget()
            self.border_width_label.pack_forget()
            self.border_width_entry.pack_forget()
        if self.filter_choice.get() != "Sobel Edge Detection":
            self.kernel_size_label.pack_forget()
            self.kernel_size_entry.pack_forget()
        if self.filter_choice.get() != "Laplacian Edge Detection":
            self.filter_choice_solution_label.pack_forget()
            self.solution_filter_dropdown.pack_forget()
        if self.filter_choice.get() != "Canny Edge Detection":
            self.low_threshold_label.pack_forget()
            self.low_threshold_entry.pack_forget()
            self.high_threshold_label.pack_forget()
            self.high_threshold_entry.pack_forget()
        if self.filter_choice.get() != "Deriche Edge Detection":
            self.alpha_label.pack_forget()
            self.alpha_entry.pack_forget()
            self.kernel_size_label.pack_forget()
            self.kernel_size_entry.pack_forget()
        if self.filter_choice.get() != "Harris Corner Detection":
            self.corner_quality_label.pack_forget()
            self.corner_quality_entry.pack_forget()
            self.min_distance_label.pack_forget()
            self.min_distance_entry.pack_forget()
            self.block_size_label.pack_forget()
            self.block_size_entry.pack_forget()
        if self.filter_choice.get() != "Cascade Classifier":
            self.conf_choice_label.pack_forget()
            self.conf_choice_dropdown.pack_forget()
        if self.filter_choice.get() != "Contour Detection":
            self.border_color_label.pack_forget()
            self.border_color_entry.pack_forget()
            self.border_width_label.pack_forget()
            self.border_width_entry.pack_forget()
        if self.filter_choice.get() != "Watershed":
            self.border_color_label.pack_forget()
            self.border_color_entry.pack_forget()
            self.border_width_label.pack_forget()
            self.border_width_entry.pack_forget()


    # Burada yapılan şey seçili olan filtrelen hide_params fonksiyonu ile gizlenmiş olan parametresini
    # Eğer o filtre seçili ise görünür yapması
    def show_params(self, selected_filter):
        self.hide_params()
        if selected_filter == "Adaptive Threshold":
            self.param_label_1.pack(padx=5, pady=5)
            self.block_size_entry.pack(padx=5, pady=5)
            self.param_label_2.pack(padx=5, pady=5)
            self.c_entry.pack(padx=5, pady=5)
        elif selected_filter == "Add-Border":
            self.border_width_label.pack(padx=5, pady=5)
            self.border_width_entry.pack(padx=5, pady=5)
            self.border_color_label.pack(padx=5, pady=5)
            self.border_color_entry.pack(padx=5, pady=5)
        elif selected_filter == "Blur":
            self.filter_choice_blur_label.pack(padx=5, pady=5)
            self.blur_filter_dropdown.pack(padx=5, pady=5)
            self.kernel_size_label.pack(padx=5, pady=5)
            self.kernel_size_entry.pack(padx=5, pady=5)
        elif selected_filter == "Sharpen":
            self.filter_choice_sharpen_label.pack(padx=5, pady=5)
            self.sharpen_filter_dropdown.pack(padx=5, pady=5)
        elif selected_filter == "Gamma Fix":
            self.gamma_value_label.pack(padx=5, pady=5)
            self.gamma_value_entry.pack(padx=5, pady=5)
        elif selected_filter == "Sobel Edge Detection":
            self.kernel_size_label.pack(padx=5, pady=5)
            self.kernel_size_entry.pack(padx=5, pady=5)
        elif selected_filter == "Laplacian Edge Detection":
            self.filter_choice_solution_label.pack(padx=5, pady=5)
            self.solution_filter_dropdown.pack(padx=5, pady=5)
        elif selected_filter == "Canny Edge Detection":
            self.low_threshold_label.pack(padx=5, pady=5)
            self.low_threshold_entry.pack(padx=5, pady=5)
            self.high_threshold_label.pack(padx=5, pady=5)
            self.high_threshold_entry.pack(padx=5, pady=5)
        elif selected_filter == "Deriche Edge Detection":
            self.alpha_label.pack(padx=5, pady=5)
            self.alpha_entry.pack(padx=5, pady=5)
            self.kernel_size_label.pack(padx=5, pady=5)
            self.kernel_size_entry.pack(padx=5, pady=5)
        elif selected_filter == "Harris Corner Detection":
            self.corner_quality_label.pack(padx=5, pady=5)
            self.corner_quality_entry.pack(padx=5, pady=5)
            self.min_distance_label.pack(padx=5, pady=5)
            self.min_distance_entry.pack(padx=5, pady=5)
            self.block_size_label.pack(padx=5, pady=5)
            self.block_size_entry.pack(padx=5, pady=5)
        elif selected_filter == "Cascade Classifier":
            self.conf_choice_label.pack(padx=5, pady=5)
            self.conf_choice_dropdown.pack(padx=5, pady=5)
        elif selected_filter == "Contour Detection":
            self.border_width_label.pack(padx=5, pady=5)
            self.border_width_entry.pack(padx=5, pady=5)
            self.border_color_label.pack(padx=5, pady=5)
            self.border_color_entry.pack(padx=5, pady=5)
        elif selected_filter == "Watershed":
            self.border_width_label.pack(padx=5, pady=5)
            self.border_width_entry.pack(padx=5, pady=5)
            self.border_color_label.pack(padx=5, pady=5)
            self.border_color_entry.pack(padx=5, pady=5)

    def open_file_dialog(self):
        # Dosya açma iletişim kutusunu açar ve kullanıcıdan resim dosyası seçmesini ister
        self.file_path = filedialog.askopenfilename(filetypes=[('Image files', '*.jpg;*.jpeg;*.png;*.gif')])
        if self.file_path:  # Eğer bir dosya seçildiyse devam eder
            self.original_img = Image.open(self.file_path)  # Seçilen dosyayı PIL kütüphanesi ile açar
            resized_img = self.original_img.resize((400, 400), Image.LANCZOS)  # Resmi boyutlandırır
            photo = ImageTk.PhotoImage(resized_img)  # Tkinter için uygun formata dönüştürür
            self.original_label.configure(image=photo)  # Etiketi (label) yeniden ayarlar ve resmi gösterir
            self.original_label.image = photo  # Etiketi (label) günceller

    """
    def open_file_dialog(self):
        file_path = filedialog.askopenfilename(filetypes=[('Image files', '*.jpg;*.jpeg;*.png;*.gif')])
        if file_path:
            self.original_img_cv = cv2.imread(file_path)
            resized_img_cv = cv2.resize(self.original_img_cv, (400, 400), interpolation=cv2.INTER_LANCZOS4)
            cv2.imshow('Resized Image', resized_img_cv)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    """

    # Filtre uygulama butonuna basıldıktan sonra hangi butonun seçili olduğunu kontrol eder.
    # Hangisi seçiliyse ona göre filtreleme işlemi uygular.
    def apply_filter(self):
        if self.original_img:
            selected_filter = self.filter_choice.get()

            # Bulanıklaştırma Filtresi
            if selected_filter == "Blur":
                selected_blur = self.filter_choice_blur.get()
                kernel_size = int(self.kernel_size_entry.get())
                if selected_blur == "Blur":
                    if kernel_size % 2 == 1 and kernel_size > 0:  # Kernel boyutu tek olmalı ve 0'dan büyük olmalı
                        cv_img = self.pil_to_opencv(self.original_img)
                        self.filtered_img = cv2.blur(cv_img, (kernel_size, kernel_size))
                        self.filtered_img = Image.fromarray(cv2.cvtColor(self.filtered_img, cv2.COLOR_BGR2RGB))
                    else:
                        messagebox.showerror("Hata", "Kernel boyutu pozitif ve tek olmalıdır.")

                elif selected_blur == "Median Blur":
                    if kernel_size % 2 == 1 and kernel_size > 0:  # Kernel boyutu tek olmalı ve 0'dan büyük olmalı
                        cv_img = self.pil_to_opencv(self.original_img)
                        self.filtered_img = cv2.medianBlur(cv_img, kernel_size)
                        self.filtered_img = Image.fromarray(cv2.cvtColor(self.filtered_img, cv2.COLOR_BGR2RGB))
                    else:
                        messagebox.showerror("Hata", "Kernel boyutu pozitif ve tek olmalıdır.")
                elif selected_blur == "Box Filter":
                    if kernel_size % 2 == 1 and kernel_size > 0:  # Kernel boyutu tek olmalı ve 0'dan büyük olmalı
                        cv_img = self.pil_to_opencv(self.original_img)
                        self.filtered_img = cv2.boxFilter(cv_img, -1, (kernel_size, kernel_size))
                        self.filtered_img = Image.fromarray(cv2.cvtColor(self.filtered_img, cv2.COLOR_BGR2RGB))

                    else:
                        messagebox.showerror("Hata", "Kernel boyutu pozitif ve tek olmalıdır.")
                elif selected_blur == "bilateralFilter":
                    if kernel_size % 2 == 1 and kernel_size > 0:  # Kernel boyutu tek olmalı ve 0'dan büyük olmalı
                        cv_img = self.pil_to_opencv(self.original_img)
                        self.filtered_img = cv2.bilateralFilter(cv_img, 9, 75, 75)
                        self.filtered_img = Image.fromarray(cv2.cvtColor(self.filtered_img, cv2.COLOR_BGR2RGB))

                    else:
                        messagebox.showerror("Hata", "Kernel boyutu pozitif ve tek olmalıdır.")
                elif selected_blur == "GaussianBlur":
                    if kernel_size % 2 == 1 and kernel_size > 0:  # Kernel boyutu tek olmalı ve 0'dan büyük olmalı
                        cv_img = self.pil_to_opencv(self.original_img)
                        self.filtered_img = cv2.GaussianBlur(cv_img, (kernel_size, kernel_size), 0)
                        self.filtered_img = Image.fromarray(cv2.cvtColor(self.filtered_img, cv2.COLOR_BGR2RGB))

                    else:
                        messagebox.showerror("Hata", "Kernel boyutu pozitif ve tek olmalıdır.")
            # Keskinleştirme Filtresi
            elif selected_filter == "Sharpen":
                selected_sharpen = self.filter_choice_sharpen.get()


                if selected_sharpen == "Sharpen":
                    sharpen_array = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                    cv_img = self.pil_to_opencv(self.original_img)
                    self.filtered_img = cv2.filter2D(cv_img, -1, sharpen_array)
                    self.filtered_img = Image.fromarray(cv2.cvtColor(self.filtered_img, cv2.COLOR_BGR2RGB))

                elif selected_sharpen == "Outline":
                    outline_array = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
                    cv_img = self.pil_to_opencv(self.original_img)
                    self.filtered_img = cv2.filter2D(cv_img, -1, outline_array)
                    self.filtered_img = Image.fromarray(cv2.cvtColor(self.filtered_img, cv2.COLOR_BGR2RGB))

            # Görüntüdeki nesnelerin sınırlarını bulmayı ve izole etmeyi sağlar.
            elif selected_filter == "Contour":
                if self.filtered_img is None:
                    self.filtered_img = self.original_img.filter(ImageFilter.CONTOUR)
                else:
                    self.filtered_img = self.original_img.filter(ImageFilter.CONTOUR)
            # Görüntüyü siyah ve beyaz bölgelere ayırırken, farklı bölgesel ayarlamalar yaparak daha iyi sonuçlar elde etmeyi sağlar.
            elif selected_filter == "Adaptive Threshold":
                block_size_text = self.block_size_entry.get()
                c_val_text = self.c_entry.get()

                if block_size_text.isdigit() and c_val_text.isdigit():
                    block_size = int(block_size_text)
                    c_val = int(c_val_text)

                    if block_size % 2 == 1 and block_size > 1:
                        cv_img = cv2.cvtColor(np.array(self.original_img), cv2.COLOR_RGB2BGR)
                        grayscale_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                        adaptive_threshold = cv2.adaptiveThreshold(grayscale_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                                   cv2.THRESH_BINARY, block_size, c_val)
                        self.filtered_img = Image.fromarray(adaptive_threshold)
                    else:
                        messagebox.showerror("Hata", "Blok boyutu 1'den büyük ve tek olmalıdır.")
                else:
                    messagebox.showerror("Hata", "Geçersiz giriş. Blok boyutu ve C değeri tam sayı olmalıdır.")
            # Görüntüye kenarlık ekler veya var olan kenarlık özelliklerini değiştirir.
            elif selected_filter == "Add-Border":
                border_color = self.border_color_entry.get()
                border_width = int(self.border_width_entry.get())

                if border_width > 0:
                    cv_img = np.array(self.original_img)
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
                    border_color = tuple(map(int, border_color.split(',')))
                    image_with_border = cv2.copyMakeBorder(cv_img, border_width, border_width, border_width,
                                                           border_width,

                                                           borderType=cv2.BORDER_CONSTANT, value=border_color)

                    self.filtered_img = Image.fromarray(cv2.cvtColor(image_with_border, cv2.COLOR_BGR2RGB))

                    if self.filtered_img:
                        resized_filtered_img = self.filtered_img.resize((400, 400), Image.LANCZOS)
                        photo = ImageTk.PhotoImage(resized_filtered_img)
                        self.filtered_label.configure(image=photo)
                        self.filtered_label.image = photo

                else:
                    messagebox.showerror("Hata", "Kenarlık genişliği pozitif bir tam sayı olmalıdır.")
            # Görüntü parlaklığını veya kontrastını düzenler, genellikle gamma ayarlamalarıyla yapılır.
            elif selected_filter == "Gamma Fix":
                gamma_value = float(self.gamma_value_entry.get())
                cv_img = self.pil_to_opencv(self.original_img)

                def apply_gamma_correction(image, gamma=1.0):
                    # Normalize the image pixel value to the range [0,1]
                    image_normalized = image / 255.0

                    # Apply gamma corretion
                    gamma_corrected = np.power(image_normalized, gamma)

                    # Rescale the image back to the range [0, 255]
                    gamma_corrected = np.uint8(gamma_corrected * 255)

                    return gamma_corrected

                self.filtered_img = apply_gamma_correction(cv_img, gamma=gamma_value)
                self.filtered_img = Image.fromarray(cv2.cvtColor(self.filtered_img, cv2.COLOR_BGR2RGB))
            # Görüntüdeki kenarları bulur, yatay ve dikey yöndeki gradyan değişikliklerini algılar.
            elif selected_filter == "Sobel Edge Detection":
                kernel_size = int(self.kernel_size_entry.get())
                cv_img = self.pil_to_opencv(self.original_img)
                if kernel_size % 2 == 1 and kernel_size > 0 and kernel_size < 32:  # Kernel boyutu tek olmalı ve 0'dan büyük olmalı
                    sobelx = cv2.Sobel(cv_img, cv2.CV_64F, 1, 0, ksize=kernel_size)
                    sobely = cv2.Sobel(cv_img, cv2.CV_64F, 0, 1, ksize=kernel_size)

                    sobelx = np.abs(sobelx)
                    sobely = np.abs(sobely)

                    self.filtered_img = cv2.bitwise_or(sobelx, sobely)
                    self.filtered_img = cv2.convertScaleAbs(self.filtered_img)
                    self.filtered_img = Image.fromarray(cv2.cvtColor(self.filtered_img, cv2.COLOR_BGR2RGB))
                else:
                    messagebox.showerror("Hata", "Kernel boyutu pozitif,tek ve en fazla 31 olabilir.")
            # Kenarları ve konturları tespit etmek için bir diğer filtreleme yöntemidir.
            elif selected_filter == "Laplacian Edge Detection":
                selected_solution = self.filter_choice_solution.get()
                cv_img = self.pil_to_opencv(self.original_img)
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

                if selected_solution == "Solution-1":

                    sonuc1 = cv2.Laplacian(cv_img, cv2.CV_64F)
                    sonuc1 = np.uint8(np.absolute(sonuc1))
                    self.filtered_img = sonuc1
                    self.filtered_img = Image.fromarray(cv2.cvtColor(self.filtered_img, cv2.COLOR_BGR2RGB))

                elif selected_solution == "Solution-2":
                    imgBlured = cv2.GaussianBlur(cv_img, (3, 3), 0)
                    sonuc2 = cv2.Laplacian(imgBlured, ddepth=-1, ksize=3)

                    self.filtered_img = sonuc2
                    self.filtered_img = Image.fromarray(cv2.cvtColor(self.filtered_img, cv2.COLOR_BGR2RGB))

                else:
                    messagebox.showerror("Hata", "Solution Seçilmeli.")
            # Görüntüdeki kenarları tespit etmek için çok kullanılan ve hassas bir yöntemdir.
            elif selected_filter == "Canny Edge Detection":
                selected_solution = self.filter_choice_solution.get()
                cv_img = self.pil_to_opencv(self.original_img)
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

                low_threshold_value = int(self.low_threshold_entry.get())
                high_threshold_value = int(self.high_threshold_entry.get())
                self.filtered_img = cv2.Canny(cv_img, low_threshold_value, high_threshold_value, L2gradient=True)
                self.filtered_img = Image.fromarray(cv2.cvtColor(self.filtered_img, cv2.COLOR_BGR2RGB))
            # Görüntüdeki kenarları tespit etmek için bir diğer özel bir filtreleme algoritmasıdır.
            elif selected_filter == "Deriche Edge Detection":

                alpha_value = float(self.alpha_entry.get())
                kernel_size = int(self.kernel_size_entry.get())

                if kernel_size % 2 == 0 or kernel_size > 31 or kernel_size < 0:
                    messagebox.showerror("Hata", "Kernel boyutu pozitif,tek ve en fazla 31 olabilir.")
                else:

                    cv_img = self.pil_to_opencv(self.original_img)
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

                    kx, ky = cv2.getDerivKernels(1, 1, kernel_size, normalize=True)
                    deriche_kernel_x = alpha_value * kx
                    deriche_kernel_y = alpha_value * ky

                    # Görüntüyü Deriche Filtresi ile türevle
                    deriche_x = cv2.filter2D(cv_img, -1, deriche_kernel_x)
                    deriche_y = cv2.filter2D(cv_img, -1, deriche_kernel_y)

                    edges = np.sqrt(deriche_x * 2 + deriche_y * 2)
                    self.filtered_img = edges
                    """filtreleme hassasiyeti çok düşük"""
                    self.filtered_img = self.filtered_img.astype(np.uint8)
                    self.filtered_img = Image.fromarray(cv2.cvtColor(self.filtered_img, cv2.COLOR_BGR2RGB))
            # Köşeleri ve özellikli noktaları tespit etmek için kullanılır.
            elif selected_filter == "Harris Corner Detection":
                cv_img = self.pil_to_opencv(self.original_img)

                gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                corner_quality = float(self.corner_quality_entry.get())
                min_distance = int(self.min_distance_entry.get())
                block_size = int(self.block_size_entry.get())

                corners = cv2.cornerHarris(gray, block_size, 3, corner_quality)
                corners = cv2.dilate(corners, None)
                cv_img[corners > 0.01 * corners.max()] = [0, 0, 255]

                self.filtered_img = cv_img
                self.filtered_img = Image.fromarray(cv2.cvtColor(self.filtered_img, cv2.COLOR_BGR2RGB))
            # Özellik tabanlı nesne tespiti ve sınıflandırma için kullanılan bir yöntemdir, genellikle nesne tanıma için kullanılır.
            elif selected_filter == "Cascade Classifier":
                selected_conf = self.conf_choice_solution.get()
                cv_img = self.pil_to_opencv(self.original_img)
                faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

                conf_1 = cv_img
                conf_2 = conf_1.copy()

                faces_1 = faceCascade.detectMultiScale(conf_1)
                for (x, y, w, h) in faces_1:
                    cv2.rectangle(conf_1, (x, y), (x + w, y + h), (255, 0, 0), 10)
                    faces_2 = faceCascade.detectMultiScale(conf_2, scaleFactor=1.3, minNeighbors=6)

                if selected_conf == "Conf-1":
                    self.filtered_img = conf_1
                    self.filtered_img = Image.fromarray(cv2.cvtColor(self.filtered_img, cv2.COLOR_BGR2RGB))
                elif selected_conf == "Conf-2":
                    for (x, y, w, h) in faces_2:
                        cv2.rectangle(conf_2, (x, y), (x + w, y + h), (255, 0, 0), 10)
                        self.filtered_img = conf_2
                        self.filtered_img = Image.fromarray(cv2.cvtColor(self.filtered_img, cv2.COLOR_BGR2RGB))
            # Nesnelerin sınırlarını belirlemek için kullanılır, genellikle nesne tanıma ve şekil analizi işlemlerinde kullanılır.
            elif selected_filter == "Contour Detection":
                border_color = self.border_color_entry.get()
                border_color = tuple(map(int, border_color.split(',')))
                border_width = int(self.border_width_entry.get())
                if border_width > 0:
                    cv_img = self.pil_to_opencv(self.original_img)
                    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray, 50, 150)

                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(cv_img, contours, -1, border_color, border_width)
                    self.filtered_img = cv_img
                    self.filtered_img = Image.fromarray(cv2.cvtColor(self.filtered_img, cv2.COLOR_BGR2RGB))
                else:
                    messagebox.showerror("Hata", "Kenarlık genişliği pozitif bir tam sayı olmalıdır.")
            # Nesneleri ve nesne sınırlarını analiz etmek ve bölmek için kullanılan bir algoritmadır. Özellikle segmentasyon için kullanılır.
            elif selected_filter == "Watershed":

                border_color = self.border_color_entry.get()
                border_color = tuple(map(int, border_color.split(',')))
                border_width = int(self.border_width_entry.get())
                if border_width > 0:
                    # Görüntüyü yükle
                    imgOrj = self.pil_to_opencv(self.original_img)
                    imgBlr = cv2.medianBlur(imgOrj, 31)
                    # Madeni para içindeki detayların sonucu etkilememesi için blurring yaptık
                    imgGray = cv2.cvtColor(imgBlr, cv2.COLOR_BGR2GRAY)
                    # Griye çevrilen resim THRESH_BINARY_INV ile arkaplan siyah, önplan beyaz yapılır
                    ret, imgTH = cv2.threshold(imgGray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                    kernel = np.ones((5, 5), np.uint8)
                    # imgTH çıktısında ön plan nesneleri üzerinde kalan gürültüden kurtulmak için
                    # morfolojik operatörlerden openning işlemi uygulanır.
                    imgOPN = cv2.morphologyEx(imgTH, cv2.MORPH_OPEN, kernel, iterations=7)

                    # Arkaplan alanı
                    # dilate() fonksiyonu ile nesneler genişletilir ve kesin emin olduğumuz arka plan kısımları elde edilir.
                    sureBG = cv2.dilate(imgOPN, kernel, iterations=3)

                    # ÖnPlan Alanı
                    # distanceTransform() ile her pikselin en yakın sıfır değerine sahip piksele
                    # olan mesafesi hesaplanır. Nesnelerin merkez pikselleri yani sıfır piksellerine en
                    # uzak nokta beyaz kalırken, siyah piksellere yaklaştıkça piksel değerleri düşer
                    # Böylece madeni para, yani emin olduğumuz ön plan piksellerin ortaya çıkmasını sağlar.
                    dist_transform = cv2.distanceTransform(imgOPN, cv2.DIST_L2, 5)

                    # Eşikleme yap
                    # Eşik değeri olarak hesaplanan maksimum mesafenin %70'den büyük olanlarının
                    # piksel değeri 255 yapılarak sureFG elde edilmiştir.
                    ret, sureFG = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

                    # Bilinmeyen bölgeleri bul
                    # Emin olduğumuz arkaplan ve ön plan arasında kalan alandır.
                    sureFG = np.uint8(sureFG)
                    unknown = cv2.subtract(sureBG, sureFG)

                    # Etiketleme işlemi
                    ret, markers = cv2.connectedComponents(sureFG, labels=5)

                    # Bilinmeyen pikselleri etiketle
                    markers = markers + 1
                    markers[unknown == 255] = 0
                    # Watershed algoritması uygula
                    markers = cv2.watershed(imgOrj, markers)

                    contours, hierarchy = cv2.findContours(markers, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

                    imgCopy = imgOrj.copy()
                    for i in range(len(contours)):
                        if hierarchy[0][i][3] == -1:
                            cv2.drawContours(imgCopy, contours, i, border_color, border_width)

                    self.filtered_img = imgCopy
                    self.filtered_img = Image.fromarray(cv2.cvtColor(self.filtered_img, cv2.COLOR_BGR2RGB))

            if self.filtered_img is not None:
                # Filtrelenmiş resmi yeniden boyutlandırır ve tkinter için uygun formata dönüştürür
                resized_filtered_img = self.filtered_img.resize((400, 400), Image.LANCZOS)
                photo = ImageTk.PhotoImage(resized_filtered_img)
                self.filtered_label.configure(image=photo)  # Etiketi (label) yeniden ayarlar ve filtrelenmiş resmi gösterir
                self.filtered_label.image = photo  # Etiketi (label) günceller

    def save_image(self):
        # Eğer filtrelenmiş resim mevcutsa kaydedilir
        if self.filtered_img:
            # Kullanıcıya kaydetme yolu belirtmesi için dosya iletişim kutusu açılır
            save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[('PNG files', '*.png')])
            if save_path:  # Eğer bir kayıt yolu belirlenmişse devam eder
                self.filtered_img.save(save_path)  # Filtrelenmiş resmi belirtilen yola kaydeder
                # Kullanıcıya başarılı kayıt mesajı gösterir
                messagebox.showinfo("Bilgi", "Filtrelenmiş resim kaydedildi: {}".format(save_path))


def main():
    # Tkinter penceresi oluşturulur
    root = tk.Tk()

    # ImageFilterApp sınıfından bir örnek oluşturulur
    app = ImageFilterApp(root)

    # Pencere boyutu belirlenir
    root.geometry('1050x500')

    # Uygulamanın ana döngüsü başlatılır
    root.mainloop()


# Dosya doğrudan çalıştırıldığında main() fonksiyonu çağrılır
if __name__ == '__main__':
    main()
