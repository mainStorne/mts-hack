"""Identity: the frontend generates a random id once and sends it as `X-Client-Id`. It only separates the chats of different
browsers; it is NOT authentication. Replace this dependency with real auth (a JWT, a session) when you have users."""

import re
from typing import Annotated

from fastapi import Depends, Header, HTTPException

CLIENT_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def client_id(x_client_id: Annotated[str | None, Header()] = None) -> str:
    if not x_client_id or not CLIENT_ID.match(x_client_id):
        raise HTTPException(
            401, "Send a header X-Client-Id: 8-64 letters, digits, - or _", headers={"WWW-Authenticate": "Client-Id"}
        )
    return x_client_id


ClientDep = Annotated[str, Depends(client_id)]
