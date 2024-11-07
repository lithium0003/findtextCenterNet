#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <thread>
#include <chrono>
#include <signal.h>
#include <curl/curl.h>

bool isLive = true;

size_t onReceive(char* ptr, size_t size, size_t nmemb, void* stream) {
	int64_t *count = (int64_t *)stream;
	const size_t sizes = size * nmemb;
    std::cout.write(ptr, sizes);
	*count += sizes;
	return sizes;
}

void sigpipe_handler(int unused)
{
	isLive = false;
}

int main(int argc, const char * argv[]) {
	if(argc < 2) {
		return 0;
	}

	signal(SIGPIPE, sigpipe_handler);

	CURL *curl = curl_easy_init();
	if (curl == nullptr) {
		curl_easy_cleanup(curl);
		return 1;
	}
	int64_t count = 0;
	curl_easy_setopt(curl, CURLOPT_URL, argv[1]);
	curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1);
	curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, onReceive);
	curl_easy_setopt(curl, CURLOPT_WRITEDATA, &count);

	// 通信実行
	CURLcode res = curl_easy_perform(curl);
	if (res != CURLE_OK) {
		fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
		int retry = 1000;
		while(isLive && res != CURLE_OK && (count > 0 || retry-- > 0)) {
			if(count > 0) {
				std::stringstream ss;
				ss << count << "-";
				std::cerr << "range:" << ss.str() << std::endl;
				curl_easy_setopt(curl, CURLOPT_RANGE, ss.str().c_str());
				res = curl_easy_perform(curl);
				if(res != CURLE_OK) {
					fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
					fprintf(stderr, "retry remain %d\n", retry);
					std::this_thread::sleep_for(std::chrono::milliseconds(500));
				}
			}
			else {
				res = curl_easy_perform(curl);
				if(res != CURLE_OK) {
					fprintf(stderr, "curl_easy_perform() failed: %s\n", curl_easy_strerror(res));
					fprintf(stderr, "retry remain %d\n", retry);
					std::this_thread::sleep_for(std::chrono::milliseconds(500));
				}
			}
		}
	}

	// std::cerr << "downloaded:" << count << std::endl;

	curl_easy_cleanup(curl);
	return 0;
}
