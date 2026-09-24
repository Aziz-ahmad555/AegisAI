"""Shared test helpers: behave like the real browser pages do (fetch the
page, then send the session's CSRF token with every state-changing request)."""
import app as aegis
from conftest import TEST_PASSWORD


def csrf_token(client):
    client.get("/login")                 # renders the form -> creates the session token
    with client.session_transaction() as s:
        return s["csrf_token"]


def login(client, username=None, password=None):
    return client.post("/login", data={
        "username": username if username is not None else aegis.credentials.username,
        "password": password if password is not None else TEST_PASSWORD,
        "csrf_token": csrf_token(client),
    })


def api_post(client, url, json=None):
    with client.session_transaction() as s:
        token = s.get("csrf_token")
    if token is None:
        token = csrf_token(client)
    return client.post(url, json=json, headers={"X-CSRF-Token": token})
