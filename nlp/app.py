from flask import Flask, render_template, request
import nltk
import utils

app = Flask(__name__, template_folder='templates')

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

@app.route('/')
def home():
    return render_template("index.html")


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

if __name__ == "__main__":
    app.run(debug=True)