from LSTM import predict, fetch_model
from flask import Flask, request, render_template

app = Flask(__name__)

model = None


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    global model
    if model == None:
        model = fetch_model()
    if request.method == 'POST':
        input_text = request.form['news']
        pred = predict(input_text,model)[0]
        prediction = f"{'FAKE' if pred > 0.5 else 'REAL'} ({pred:.2f})"
    return render_template('/index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

print(predict(["As was revealed by Wikileaks, Hillary Clinton spent the run up to her presidential campaign giving speeches to Goldman Sachs and other Wall Street banks, where she praised their talents and explained her positions on financial regulation.  On October 24, 2013, Clinton told Goldman Sachs that Dodd-Frank had to be done mostly for “political reasons” because Congress needed to look like it was doing something about the crisis. She said, “There’s nothing magic about regulations, too much is bad, too little is bad. How do you get to the golden key, how do we figure out what works? And the people that know the industry better than anybody are the people who work in the industry.” Yes, she essentially endorsed Wall Street writing the rules because Wall Street knows its business best and complained to Goldman Sachs that regulations had frightened bankers. “I mean, right now, there are so many places in our country where the banks are not doing what they need to do because they’re scared of regulations, they’re scared of the other shoe dropping, they’re just plain scared, so credit is not flowing the way it needs to to restart economic growth,” Clinton said. “So people are, you know, a little — they’re still uncertain, and they’re uncertain both because they don’t know what might come next in terms of regulations, but they’re also uncertain because of changes in a global economy that we’re only beginning to take hold of.” Music to Wall Street’s ears. For Goldman Sachs, was there really any other choice this cycle? After all, they did pay Hillary Clinton $675,000 for those three speeches, and have generously supported her political career. Despite her private comments to Goldman Sachs, Hillary Clinton has taken a tough public position on Wall Street during the campaign, likely due to Senator Bernie Sanders’ success in the primaries. Of course, Wikileaks also revealed that Clinton told the National Multi-housing Council in a private speech that “you need both a public and a private position.” So the real question is, what do Blankfein and Goldman want in return and what is Clinton’s private position on giving it to them? Be Sociable, Share!"]))