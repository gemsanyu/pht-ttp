import matplotlib.pyplot as plt

from ttp.ttp import TTP


def run():
    dataset_name, gname = "lin105_n104_bounded-strongly-corr_01", "lin105"
    # dataset_name, gname = "eil76-n75", "eil76"
    # dataset_name, gname = "ch150_n149_bounded-strongly-corr_01", "ch150"
    prob = TTP(dataset_name=dataset_name)
    coords = prob.location_data.coords
    marker_size = 80
    plt.scatter(coords[:,0], coords[:,1], c="white", s=marker_size, edgecolors="blue", label="cities")
    plt.scatter(coords[0,0], coords[0,1], c="goldenrod", s=marker_size, label="first city")
    plt.title(gname)
    plt.legend(fancybox=True, shadow=True)
    plt.savefig(fname=gname+".pdf", format="pdf", dpi=300)
    plt.show()
    return         

if __name__ == "__main__":
    run()
