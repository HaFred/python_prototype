from my_img2num import MyImg2Num
from nn_img2num import NnImg2Num


def main():
    # model_my = MyImg2Num()
    model_my = NnImg2Num()
    model_my.train()
    print('done')


if __name__ == "__main__":
    main()
