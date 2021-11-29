import numpy as np
import ManualStrategy as ms
import experiment1 as exp1
import experiment2 as exp2


def author():
    return 'mlukacsko3'


if __name__ == "__main__":
    ms.main()
    np.random.seed(12345)
    exp1.main()
    np.random.seed(12345)
    exp2.main()


