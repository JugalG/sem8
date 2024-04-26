def main():
    processes = int(input("number of processes : "))
    resources = int(input("number of resources : "))
    max_resources = [int(i) for i in input("maximum resources : ").split()]

    print("\n-- allocated resources for each process --")
    currently_allocated = [[int(i) for i in input(f"process {j + 1} : ").split()] for j in range(processes)]

    print("\n-- maximum resources for each process --")
    max_need = [[int(i) for i in input(f"process {j + 1} : ").split()] for j in range(processes)]

    allocated = [0] * resources
    for i in range(processes):
        for j in range(resources):
            allocated[j] += currently_allocated[i][j]
    print(f"\ntotal allocated resources : {allocated}")

    available = [max_resources[i] - allocated[i] for i in range(resources)]
    print(f"total available resources : {available}\n")

    sequence=[]

    running = [True] * processes
    count = processes
    while count != 0:
        safe = False
        for i in range(processes):
            if running[i]:
                executing = True
                for j in range(resources):
                    if max_need[i][j] - currently_allocated[i][j] > available[j]:
                        executing = False
                        break
                if executing:
                    print(f"process {i + 1} is executing")
                    running[i] = False
                    count -= 1
                    safe = True
                    for j in range(resources):
                        available[j] += currently_allocated[i][j]
                    sequence.append(i+1)
                    break
        if not safe:
            print("the processes are in an unsafe state.")
            break

        print(f"the process is in a safe state.\navailable resources : {available}\n")
    
    print("The sequence is :")
    s=""
    for i in range (processes-1):
        s+=f"P{sequence[i]} -> "
    s+=f"P{sequence[processes-1]}"

    print(s)


if __name__ == '__main__':
    main()




   
    
# Sample Input :
    
# number of processes : 5
# number of resources : 3
# maximum resources : 10 5 7
# -- allocated resources for each process --
# process 1 : 0 1 0
# process 2 : 2 0 0
# process 3 : 3 0 2
# process 4 : 2 1 1
# process 5 : 0 0 2
# -- maximum resources for each process --
# process 1 : 7 5 3
# process 2 : 3 2 2
# process 3 : 9 0 2
# process 4 : 2 2 2
# process 5 : 4 3 3

# This input represents:

# 5 processes
# 3 types of resources
# Maximum resources available: 10 of resource type 1, 5 of type 2, and 7 of type 3
# Allocation of resources for each process
# Maximum resources needed for each process
