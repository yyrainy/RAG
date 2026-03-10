import config
import tools
from index import gr_show
def main():
    config.vectordb= tools.load_exist_vector_db()

    gr_show()


# http://127.0.0.1:7860
# http://localhost:7860
if __name__ == "__main__":
    main()
