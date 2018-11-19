from hyperspace.kepler import load_results


def main():
    res = load_results('results')
    func_vals = [len(x.func_vals) for x in res]
    print(func_vals)


if __name__=='__main__':
    main()
