from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import gradio as gr

# --------- Load model và tokenizer ----------
model = load_model('model_nplm.h5')
model.summary()

# Đọc các dòng văn bản để tạo tokenizer
with open('kieu.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Tokenizer với out-of-vocabulary token
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(lines)

# Các tham số cần thiết (giống lúc train)
max_length = 5

# --------- Hàm dự đoán ----------
def predict_next_word_gradio(text, num_predictions=3):
    try:
        # Tiền xử lý: tokenize và pad
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=7, padding='post')  # Đảm bảo maxlen là 7 để phù hợp với mô hình

        predicted_words = []  # Danh sách để lưu các từ dự đoán
        
        for _ in range(num_predictions):
            # Dự đoán từ tiếp theo
            prediction = model.predict(padded)

            # Kiểm tra shape của prediction và xử lý
            if len(prediction.shape) == 3:
                # Nếu là (batch_size, max_length, vocab_size), chọn từ tiếp theo sau max_length
                predicted_index = np.argmax(prediction[0][-1])
            elif len(prediction.shape) == 2:
                # Nếu là (batch_size, vocab_size), chọn từ có xác suất cao nhất
                predicted_index = np.argmax(prediction[0])
            else:
                return "Lỗi: prediction shape không hợp lệ"

            # Lấy từ dự đoán từ index
            predicted_word = tokenizer.index_word.get(predicted_index, "<Unknown>")
            predicted_words.append(predicted_word)

            # Cập nhật đầu vào với từ dự đoán để dự đoán từ tiếp theo
            seq[0].append(predicted_index)
            padded = pad_sequences(seq, maxlen=7, padding='post')

        return f"Dự đoán các từ tiếp theo: {' '.join(predicted_words)}"
    
    except Exception as e:
        return f"Lỗi: {str(e)}"


# --------- Tạo giao diện Gradio ----------
iface = gr.Interface(
    fn=predict_next_word_gradio,
    inputs=[gr.Textbox(lines=2, placeholder="Nhập câu bất kỳ..."), gr.Slider(minimum=1, maximum=5, step=1, value=3, label="Số từ dự đoán")],
    outputs="text",
    title="Dự đoán từ tiếp theo:",
    description="Nhập một đoạn văn."
)

# Chạy giao diện Gradio
iface.launch()
