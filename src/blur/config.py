from pydantic_settings import BaseSettings


class Config(BaseSettings):
    crop_save_dir: str = '/data/crops'
    tmp_dir: str = '/dev/shm'
    detect_url: str = 'http://localhost:8001'
    model_name: str = "Speedy Gonzales"

settings = Config()
