import os

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.responses import RedirectResponse
from fastapi.security import APIKeyHeader
from langserve import add_routes
from starlette.status import HTTP_401_UNAUTHORIZED

from app.aicv import chain as pinecone_cv_chain
from app.rolecover import chain as pinecone_rolecover_chain
from app.rolematch import chain as pinecone_rolematch_chain

load_dotenv()

# Keys
CV_LANGSERVE_API_KEY = os.environ['CV_LANGSERVE_API_KEY']
API_KEYS = [CV_LANGSERVE_API_KEY]
api_key_header = APIKeyHeader(name='x-api-key', auto_error=False)


async def get_host(request: Request):
	return request.client.host


async def api_key_auth(
	api_key: str = Security(api_key_header), host: str = Depends(get_host)
):
	if host != '127.0.0.1' and api_key not in API_KEYS:
		raise HTTPException(
			status_code=HTTP_401_UNAUTHORIZED, detail='Missing or invalid API Key'
		)


app = FastAPI()


@app.get('/', dependencies=[Depends(api_key_auth)])
async def redirect_root_to_docs():
	return RedirectResponse('/docs')


add_routes(app, pinecone_cv_chain, path='/cv', dependencies=[Depends(api_key_auth)])
add_routes(
	app, pinecone_rolematch_chain, path='/match', dependencies=[Depends(api_key_auth)]
)
add_routes(
	app, pinecone_rolecover_chain, path='/cover', dependencies=[Depends(api_key_auth)]
)

# if __name__ == '__main__':
# 	import uvicorn

# 	uvicorn.run(app, host='127.0.0.1', port=8000)
