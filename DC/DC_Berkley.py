def time(s):
    ans = s.split(":")
    return int(ans[0]) * 60 + int(ans[1]) + int(ans[2])/60

slaves=int(input("Enter the number of clients : "))

master = input("Enter the Time of master node : ")
masterTime=time(master)
print(masterTime)

ls = []

for i in range(slaves):
    print(f"Enter the clock time for slave {i + 1} : ")
    x = input()
    print(time(x))
    ls.append(time(x))

def get_time(val):
    hours = int(val // 60)
    mins = int(val - hours * 60)
    seconds = int((val - int(val)) * 60)
    return str(hours) + ":" +  str(mins) + ":" + str(seconds)

def berkeley_sync(master_time, client_times):
    clients = [time - master_time for time in client_times]
    print(clients)
    avg_time = sum(clients) / len(client_times)
    print(avg_time)

    masterAdjustedTime = master_time + avg_time

    adjusted_times = [masterAdjustedTime - time for time in client_times]

    print(masterAdjustedTime)
    print(get_time(masterAdjustedTime))

    return adjusted_times, masterAdjustedTime

ans, masterAdjustedTime = berkeley_sync(masterTime, ls)
print(ans)
for i in range(slaves):
    print(f"Time for slave {i + 1} : {get_time(ls[i] + ans[i])}")

print(f"Time of master node : {get_time(masterAdjustedTime)}")

# Enter the number of clients: 3
# Enter the Time of master node (format: hh:mm:ss): 10:00:00
# Enter the clock time for slave 1 (format: hh:mm:ss): 10:00:05
# Enter the clock time for slave 2 (format: hh:mm:ss): 10:00:10
# Enter the clock time for slave 3 (format: hh:mm:ss): 10:00:15
# This will simulate three slave nodes with clock times 5 seconds, 10 seconds, and 15 seconds ahead of the master node, which has a time of 10:00:00.