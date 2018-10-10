from flask import Flask, render_template, url_for, flash, redirect
from forms import RegisterForm, LoginForm

app = Flask(__name__)
app.config["SECRET_KEY"] = "9d0f5ca111fbb387afab2a29e6aa41f3"

posts = [
    {
        "author": "Erik",
        "title": "Blog post 1",
        "date_posted": "2018-10-10",
        "content": "Testing 1"
    },
    {
        "author": "Palma",
        "title": "Blog 2",
        "date_posted": "2018-10-09",
        "content": "test2"
    }
]

@app.route("/")
@app.route("/home")
def home():
    return render_template("home.html", posts=posts, title="Home")

@app.route("/about")
def about():
    return render_template("about.html", title="About")


@app.route("/register", methods=["GET", "POST"])
def register():
    form = RegisterForm()

    if form.validate_on_submit():
        flash(f"Account created for {form.username.data}!", "success")
        return redirect(url_for("home"))

    return render_template("register.html", form=form, title="Register")

@app.route("/login", methods=["GET", "POST"])
def login():
    form=LoginForm()
    if form.validate_on_submit():
        if form.email.data == "admin@blog.com" and form.password.data == "test":
            flash("You have been logged in!", "success")
            return redirect(url_for("home"))
        else:
            flash('Login Unsuccessful. Please check username and password', 'danger')
    return render_template("login.html", form=form, title="Login")


if __name__ == "__main__":
    app.run(debug=True)
