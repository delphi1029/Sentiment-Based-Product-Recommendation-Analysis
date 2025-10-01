from flask import Flask, render_template, request
import model

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend_top5():
    print(request.method)
    user_name = request.form['User Name']
    print('User name=', user_name)
    get_top5 = model.product_recommendations_user(user_name)
    if isinstance(get_top5, str):
        return render_template('index.html', text=get_top5)
    return render_template('index.html', column_names=get_top5.columns.values,
                           row_data=list(get_top5.values.tolist()), zip=zip, text='Recommended products')


if __name__ == '__main__':
    app.run(debug=True)