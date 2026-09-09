import os
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("AEGISAI_SECRET_KEY", "aegisai-command-center-demo-key")

OPERATOR_USERNAME = os.environ.get("AEGISAI_USERNAME", "operator")
OPERATOR_PASSWORD = os.environ.get("AEGISAI_PASSWORD", "aegisai2026")


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == OPERATOR_USERNAME and password == OPERATOR_PASSWORD:
            session["logged_in"] = True
            return redirect(url_for("overview"))
        error = "Invalid credentials"
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    return redirect(url_for("login"))


@app.route("/")
@login_required
def overview():
    return render_template("overview.html")


if __name__ == "__main__":
    print("AegisAI Command Center")
    print(f"Login with username '{OPERATOR_USERNAME}' (set AEGISAI_USERNAME/AEGISAI_PASSWORD env vars to change)")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(host="127.0.0.1", port=5000, debug=True)
