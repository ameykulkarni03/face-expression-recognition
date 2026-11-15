import gradio as gr
from Hackathon_setup import face_recognition, exp_recognition

# gr.Image(type="numpy") gives an RGB uint8 array (H, W, 3) — same format
# main.py (the original FastAPI app) passed into these functions.


def similarity(img1, img2):
    if img1 is None or img2 is None:
        return "Please upload both images"
    score = face_recognition.get_similarity(img1, img2)
    return f"Dissimilarity score: {score:.3f}  (lower = more similar)"


def recognise(img):
    if img is None:
        return "Please upload an image"
    return face_recognition.get_face_class(img)


def expression(img):
    if img is None:
        return "Please upload an image"
    return exp_recognition.get_expression(img)


with gr.Blocks(title="Recognition Application") as demo:
    gr.Markdown("# Recognition Application")

    with gr.Tab("Face Similarity"):
        with gr.Row():
            img_a = gr.Image(type="numpy", label="Image 1", sources=["upload", "webcam"])
            img_b = gr.Image(type="numpy", label="Image 2", sources=["upload", "webcam"])
        sim_out = gr.Textbox(label="Result")
        gr.Button("Compare").click(similarity, [img_a, img_b], sim_out)

    with gr.Tab("Face Recognition"):
        img_c = gr.Image(type="numpy", label="Face image", sources=["upload", "webcam"])
        rec_out = gr.Textbox(label="Predicted person")
        gr.Button("Recognise").click(recognise, img_c, rec_out)

    with gr.Tab("Expression Recognition"):
        img_d = gr.Image(type="numpy", label="Face image", sources=["upload", "webcam"])
        expr_out = gr.Textbox(label="Predicted expression")
        gr.Button("Predict").click(expression, img_d, expr_out)

demo.launch()
