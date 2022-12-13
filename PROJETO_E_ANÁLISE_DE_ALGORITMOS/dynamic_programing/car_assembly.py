def assembly(a, t, e, x):
    n = len(a[0])
    T1 = [0 for i in range(n)]
    T2 = [0 for i in range(n)]

    # adding base case times
    T1[0] = e[0] + a[0][0]
    T2[0] = e[1] + a[1][0]
    # Filling the dp tables T1[] and T2[] using recursive relations
    for i in range(1, n):
        T1[i] = min(T1[i - 1] + a[0][i], T2[i - 1] + t[1][i] + a[0][i])
        T2[i] = min(T2[i - 1] + a[1][i], T1[i - 1] + t[0][i] + a[1][i])

    # finding final times and returning the minimum value
    return min(T1[n - 1] + x[0], T2[n - 1] + x[1])


from random import randint


class Station:
    def __init__(self, identifier, station_cost, transfer_cost):
        self.identifier = identifier
        self.station_cost = station_cost
        self.transfer_cost = transfer_cost

    def show(self):
        print("-> %i [%i] (%s) " % (self.station_cost, self.transfer_cost, self.identifier), end=' ')


class Line:
    # The assembly line has an entry and exit costs too.
    def __init__(self, name="", entry_cost=0, exit_cost=0, stations=None):
        self.name = name
        self.entry_cost = entry_cost
        self.exit_cost = exit_cost
        self.stations = stations

    def show(self):
        print("%s\n-> %i (entry)" % (self.name, self.entry_cost), end=' ')
        for stn in self.stations:
            stn.show()
        print("-> %i (exit)" % self.exit_cost, end='\n\n')


# This optimization method consists in run through the costs list, accumulating one by one,
# and comparing the cost of the current 'i' list position with the alternative costs,
# that is, when we can change the lines to get a smaller cost.
def optimize(line1, line2):
    # 'costs1' will store the accumulated costs of line 1.
    costs1 = []
    # Analogue to 'costs2'.
    costs2 = []
    # Stores the cheapest stations.
    cheapest_stations = []
    # This method returns this Line object that contains the complete cheapest path.
    cheapest_line = Line(name="Cheapest assembly path")

    for l_station in line1.stations:
        costs1.append(l_station.station_cost)

    for l_station in line2.stations:
        costs2.append(l_station.station_cost)

    # Adds the line entry to the cost of the first station.
    costs1[0] = line1.entry_cost + line1.stations[0].station_cost
    costs2[0] = line2.entry_cost + line2.stations[0].station_cost

    # Stores the cheapest path between cost 1 and 2 on our path list and in our Line object.
    if costs1[0] <= costs2[0]:
        cheapest_line.entry_cost = line1.entry_cost
        cheapest_stations.append(line1.stations[0])
    else:
        cheapest_line.entry_cost = line2.entry_cost
        cheapest_stations.append(line2.stations[0])

    # Calculates the current cost position in our list considering the cost between just jump to the
    # next station on the current line or do a transfer to another assembly line.
    for j in range(1, len(line1.stations), 1):
        costs1[j] = min(costs1[j-1] + costs1[j],
                        costs1[j-1] + line2.stations[j-1].transfer_cost + costs2[j])

        costs2[j] = min(costs2[j-1] + costs2[j],
                        costs2[j-1] + line1.stations[j-1].transfer_cost + costs1[j])

        if costs1[j] <= costs2[j]:
            cheapest_stations.append(line1.stations[j])
        else:
            cheapest_stations.append(line2.stations[j])

    # Defines the total - and minimal - cost considering the exit cost of the assembly lines.
    temp1 = costs1[-1] + line1.exit_cost
    temp2 = costs2[-1] + line2.exit_cost

    if temp1 <= temp2:
        total_cost = temp1
        cheapest_line.exit_cost = line1.exit_cost
    else:
        total_cost = temp2
        cheapest_line.exit_cost = line2.exit_cost

    cheapest_line.stations = cheapest_stations

    return total_cost, cheapest_line  # Returns the total cost calculated and the best assembly path.


if __name__ == "__main__":
    lines = []  # List of assembly lines.

    n = int(input("Number of stations for each assembly line\n> "))

    # Construction of our assembly lines.
    it = 0
    while it < 2:
        stns = []

        for i in range(n):
            if it == 0:
                stn_name = "A"
            else:
                stn_name = "B"

            stns.append(Station(stn_name + str(i), randint(1, 5), randint(1, 5)))

        lines.append(Line("Line " + str(it+1), randint(1, 5), randint(1, 5), stns))

        it += 1

    # Calls the optimization method.
    min_cost, cheapest_path = optimize(lines[0], lines[1])

    # Show the results.
    print("\nFor each station: station_cost [transfer_cost] [identifier]", end='\n\n')
    lines.append(cheapest_path)
    for line in lines:
        line.show()
    print("Minimum cost: %i" % min_cost)


if __name__ == "__main__":
    a = [[5, 4, 3], [2, 3, 7]]
    t = [[0, 2, 2], [0, 1, 1]]
    e = [3, 2]
    x = [3, 4]
    print("The minimum time taken is:", assembly(a, t, e, x))
