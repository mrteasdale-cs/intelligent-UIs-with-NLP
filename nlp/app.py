from flask import Flask, render_template, request, send_file, Response
import nltk
import utils
from werkzeug.wsgi import FileWrapper

app = Flask(__name__, template_folder='templates')

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/info_page')
def info_page():
    return render_template("info-extraction-ner.html")

@app.route('/text-analysis', methods=['GET', 'POST'])
def text_analysis():
    cleaned_text = None
    sentiment = None
    summary = None
    if request.method == 'POST':
        get_form_type = request.form.get('form_type')
        if get_form_type == 'sentiment':
            get_raw_text = request.form['text']
            cleaned_text = utils.preprocessing(get_raw_text)
            # analyse sentiment - calls the function from utils.py
            sentiment = utils.analyse_sentiment(cleaned_text)

        elif get_form_type == 'summary':
            get_raw_text = request.form['summary_text']
            number_sentences = int(request.form.get('summary_length'))
            summary = utils.summarise_text(get_raw_text, number_sentences)

        return render_template('text-analysis.html',
                               cleaned_text=cleaned_text,
                               sentiment=sentiment,
                               summary=summary)

    return render_template('text-analysis.html')

@app.route('/visualisations', methods=['GET', 'POST'])
def visualisations():
    text = None
    if request.method == 'POST':
        get_form_type = request.form.get('form_type')
        if get_form_type == 'wordcloud':
            text = request.form['text']
    return render_template('visualisations.html', text=text)

@app.route('/wordcloud_image')
def wordcloud_image():
    text = request.args.get('text', '')
    if not text:
        # Optionally, return a placeholder image or error
        return 'Error Generating', 404
    img = utils.generate_wordcloud(text)
    img.seek(0)

    return Response(FileWrapper(img), mimetype='image/png')


if __name__ == "__main__":
    app.run(debug=True)