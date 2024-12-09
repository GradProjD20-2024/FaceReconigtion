class DefaultKeys:
    # shared default config for all apps
    APP__NAME = "APP.NAME"
    APP__ENV = "APP.ENV"
    APP__REST_PORT = "APP.REST_PORT"
    APP__UVICORN_WORKERS = "UVICORN_WORKERS"

    MILVUS__HOST = "MILVUS_HOST"
    MILVUS__PORT = "MILVUS_PORT"
    MILVUS__COLLECTION = "MILVUS_COLLECTION"

class DefaultValues:
    # default values for app config keys
    APP_ENV_PROD = "PROD"
    APP_ENV_DEBUG = "DEBUG"
    APP_DEFAULT_NAME = "app"
    APP_DEFAULT_CONFIG_FILE = "config/settings.toml"
    APP_DEFAULT_REST_PORT = 8001

    SIMILARITY_THRESHOLD = .75