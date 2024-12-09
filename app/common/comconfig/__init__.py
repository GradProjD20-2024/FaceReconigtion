from ..comconstants import DefaultValues
from dynaconf import Dynaconf

settings = Dynaconf(
    settings_files=DefaultValues.APP_DEFAULT_CONFIG_FILE,
    envvar_prefix=False,
    load_dotenv=True,

)

# print vars to debug
# print(settings.as_dict())
