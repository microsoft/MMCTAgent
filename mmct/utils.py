import jwt
import datetime
from fastapi.responses import JSONResponse
import hashlib
from jwt import ExpiredSignatureError, InvalidSignatureError
from fastapi.exceptions import HTTPException
from azure.identity import DefaultAzureCredential,get_bearer_token_provider
from functools import wraps
import json
import ast
from dotenv import load_dotenv
load_dotenv(override=True)
import asyncio
import os
from openai import AzureOpenAI

scope = "https://cognitiveservices.azure.com/.default"
credential = get_bearer_token_provider(DefaultAzureCredential(),scope)
api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
model_name =  os.environ.get("AZURE_OPENAI_MODEL")
model_version = os.environ.get("AZURE_OPENAI_MODEL_VERSION")  # Ensure this is a valid model version
deployment_name = model_name
endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")

client = AzureOpenAI(
    azure_endpoint=endpoint,
    azure_ad_token_provider=credential,
    api_version=api_version,
)


async def is_valid_ast_literal(s):
    try:
        ast.literal_eval(s)
        return True
    except Exception as e:
        return False
    
async def openai_output_to_json(data):
    try:
        data = data[data.index("{"): data.rfind("}")+1]
        data = data.replace("`","")
        data = data.rstrip()
        data = data.rstrip("\n")
        if await is_valid_ast_literal(data):
            data = ast.literal_eval(data)
        else:
            data = json.loads(data)

        return data 
    except Exception as e:
        raise Exception(e)


def get_jwt_token(email):
    return jwt.encode(
        {
            "email": email,
            "exp": datetime.datetime.now(tz=datetime.timezone.utc) + datetime.timedelta(hours=int(os.getenv('JWT_EXPIRATION_TIME'))),
        },
        key=os.getenv('JWT_SECRET_KEY'),
        algorithm=os.getenv('JWT_ALGORITHM')
    )


def generate_sha256_hash(input_string: str) -> str:
    """
    Generate a SHA-256 hash from the input string.
    """
    # Encode the input string to bytes, then create a hash object
    sha256_hash = hashlib.sha256(input_string.encode())
    # Return the hexadecimal representation of the hash
    return sha256_hash.hexdigest()


def verify_sha256_hash(input_string: str, hash_to_check: str) -> bool:
    """
    Verify if the SHA-256 hash of the input string matches the given hash.
    """
    # Generate the hash of the input string
    computed_hash = generate_sha256_hash(input_string)
    # Compare the generated hash to the given hash
    return computed_hash == hash_to_check

async def decode_token(token):
    return jwt.decode(token, key=os.getenv('JWT_SECRET_KEY'),
        algorithms=[os.getenv('JWT_ALGORITHM')])

with open("cred.json","r") as f:
    creds = json.load(f)

def token_required(f):
    @wraps(f)
    async def decorated_function(*args, **kwargs):
        try:
            token = kwargs.get('Authorization')
            if not token:
                return JSONResponse(status_code=401, content={"message": "Token is missing!"})
            
            content = await decode_token(token)  # Assuming this function is defined

            if "email" in content:
                email_hash = generate_sha256_hash(content["email"])  # Assuming this function is defined
                if email_hash in creds.keys():  # Assuming `creds` is accessible
                    return await f(*args, **kwargs)
                else:
                    return JSONResponse(status_code=401, content={"message":"Invalid token!"})
        except ExpiredSignatureError:
            return JSONResponse(status_code=401, content={"message": "JWT Signature has expired. Kindly login again!"})
        except InvalidSignatureError:
            print("Invalid")
            return JSONResponse(status_code=401,content={"message": "JWT Signature is invalid"})
        except Exception as e:
            return JSONResponse(status_code=400,content={"message": f"Error Occured: {e}"})
    return decorated_function

async def translate_query(text,target='english'):
    try:
        json_format = {"translated_text": "Translated text from Source to Traget Language"}
        system_prompt=f''' 
            You are a Translation Agent GPT. Your job is to find all the details from the input text and translate the same from the source to the {target} language.
            Mention only the english name or the text into the response, if the text is mention in the text is in hindi or any other language indic language say odia, telugu, bhojpuri etc that is agricultural domain specific then dont convert them into english language.
            Specificities that you have to find and given in the response:
            1. Detect the source langauge and translate it to the {target} language. The source language can be a mix of language or transliteration of it.
            2. Domain Specific Terminology and its translation in english in brackets. 
            3. Specific Variety of species(e.g. IPA 15-06, IPL 203, IPH 15 03 etc) on which they are talking.
            If transcript does not contains any species or variety of species then translate the complete sentence in the {target} language.
            Make sure to add the english translated name of species and their variety if present.
            Only when sure then only add the name of species or variety of species.
            Provide the final response into the below given json: Json format: {json_format}
                Note: Dont provide ```json in the response.
            '''
        prompt = [{"role":"system","content":system_prompt},
                            {"role":"user","content":[
                                {"type": "text", "text": f"The input text transcription is: {text}"}
                                                    ],
                            }]
        response = client.chat.completions.create(
                model=model_name,
                messages=prompt,
                temperature=0,
                response_format={"type": "json_object"}
            )
        return (await openai_output_to_json(response.choices[0].message.content))['translated_text']
        
    except Exception as e:
        raise Exception(e)
    

if __name__=='__main__':
    print("Text:",asyncio.run(translate_query(text="Explain step-by-step the procedure for preparing the chilly nursery bed.")))