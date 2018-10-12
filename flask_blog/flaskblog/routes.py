from flask import render_template, url_for, flash, redirect, request
from flaskblog import app, bcrypt, db
from flaskblog.forms import RegisterForm, LoginForm
from flaskblog.models import User, Post
from flask_login import login_user, current_user, logout_user, login_required

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
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    form = RegisterForm()

    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data).decode("utf-8")
        user = User(username=form.username.data, email=form.email.data,
                    password=hashed_password)

        db.session.add(user)
        db.session.commit()

        flash(f"Your account has been created! You can now login", "success")
        return redirect(url_for("login"))

    return render_template("register.html", form=form, title="Register")

@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("home"))

    form=LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()

        if user and bcrypt.check_password_hash(user.password, form.password.data):
            login_user(user, remember=form.remember.data)
            next_page = request.args.get("next")
            if next_page:
                redirect(next_page)
            else:
                return redirect(url_for("home"))
                
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')

    return render_template("login.html", form=form, title="Login")

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("home"))

@app.route("/account")
@login_required
def account():
    return render_template("account.html", title="Account")