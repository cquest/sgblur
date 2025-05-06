from pydantic_settings import BaseSettings


class Config(BaseSettings):
    crop_save_dir: str = '/data/crops'
    tmp_dir: str = '/dev/shm'
    detect_url: str = 'http://localhost:8001'
    api_name: str = "SGBlur" # Note that this is important not to change this value after the inital setup, it will be used to handle the semantic tags updates.

settings = Config()
