#ifndef MY_TIME_UTILITY
#define MY_TIME_UTILITY
#include <Windows.h>
#include <iostream>
double getCPUTime();


class CpuTime{
	FILETIME start[4];
	FILETIME end[4];
public:
	void startTimer(){
		if (GetProcessTimes(GetCurrentProcess(), &start[0], &start[1], &start[2], &start[3]) == 0){
			std::cerr << "error call getproctime" << std::endl;
		}
	}
	void endTimer(){
		if (GetProcessTimes(GetCurrentProcess(), &end[0], &end[1], &end[2], &end[3]) == 0){
			std::cerr << "error call getproctime" << std::endl;
		}
	}
	void printTimeConsume(char* comment){
		SYSTEMTIME s;
		SYSTEMTIME e;
		if (FileTimeToSystemTime(&start[0], &s) == 0 || FileTimeToSystemTime(&end[0], &e) == 0) {
			std::cerr << "error converting start time" << std::endl;
		}
		if (s.wHour != e.wHour || s.wMinute != e.wMinute || s.wMilliseconds != e.wMilliseconds) {
			std::cout << "opp!" << std::endl;
		}
		if (!FileTimeToSystemTime(&start[2], &s) || !FileTimeToSystemTime(&end[2], &e)) {
			std::cerr << "error converting kernel time" << std::endl;
		}
		double kerneltime = toSeconds(e) - toSeconds(s);
		if (!FileTimeToSystemTime(&start[3], &s) || !FileTimeToSystemTime(&end[3], &e)){
			std::cerr << "error conv user time" << std::endl;
		}
		double usertime = toSeconds(e) - toSeconds(s);
		std::cout << comment << std::endl;
		std::cout << "spent " << kerneltime << "s in kernel, " << usertime << " s in user space" << std::endl;
	}
	double toSeconds(SYSTEMTIME& t){
		return (double)t.wHour * 3600 + (double)t.wMinute * 60 + (double)t.wSecond + (double)t.wMilliseconds / 1000;;
	}
	void stopAndPrint(char* comment)
	{
		endTimer();
		printTimeConsume(comment);
	}
};
#endif // !MY_TIME_UTILITY
