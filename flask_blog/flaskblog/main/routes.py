from flask import render_template, Blueprint


main = Blueprint("main", __name__)


posts = [
    {
        "author": "Erik",
        "title": "First blog post",
        "date_posted": "2018-10-10",
        "content": "Test commit"
    }
]


@main.route("/")
@main.route("/home")
def home():
    return render_template("home.html", posts=posts, title="Home")


@main.route("/about")
def about():
    return render_template("about.html", title="About")
