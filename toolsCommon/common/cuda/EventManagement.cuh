/*
 * EventManagement.cuh
 *
 *  Created on: April 03, 2016
 *		Author: Haiyang Li
 *  Copyright Haiyang Li. All Rights Reserved.
 */

#ifndef EVENTMANAGEMENT_CUH_
#define EVENTMANAGEMENT_CUH_

#include <vector>
#include <string>
#include <sstream>

#include <cuda_runtime.h>

/**
 * Wrapper class to perform successive time measurements using CUDA events
 */
class EventRecord
{
public:
    /**
     * Constructor creating an initial number of events
     * for best performance make sure the constructor creates
     * a sufficient number of events. Dynamic creation can impact
     * execution performance
     * @param supported initial number of events to be generated
     */
    EventRecord(int supported = 8) : events_(supported){
		for (int i = 0; i < events_.size(); ++i){
			cudaEventCreate(&events_[i]);
        }
    }

    /**
     * Destructor destroying all events
     */
    ~EventRecord(){
		for (int i = 0; i < events_.size(); ++i){
			cudaEventDestroy(events_[i]);
        }
    }

    /**
     * addRecords inserts the next event into the CUDA execution pipeline
     * @param name optional name to be used in the report
     */
    void addRecord(const std::string& name = std::string()){
		if (names_.size() >= events_.size()){
			events_.resize(2 * events_.size());
			for (int i = events_.size() / 2; i < events_.size(); ++i){
				cudaEventCreate(&events_[i]);
            }
        }
		cudaEventRecord(events_[names_.size()]);

        if (name.size() == 0){
            std::stringstream sstr;
			sstr << "Event " << names_.size();
			names_.push_back(sstr.str());
        }
        else
			names_.push_back(name);
    }

    /**
     * print prints the event statistics for all captured events
     * Note that the method performs a cudaDeviceSynchronize to
     * make sure the GPU execution has finished
     */
    void print() const{
        cudaDeviceSynchronize();
		for (int i = 0; i < names_.size() - 1; ++i) {
            float ms;
			cudaEventElapsedTime(&ms, events_[i], events_[i + 1]);
			printf("%s->%s took %fms\n", names_[i].c_str(), names_[i + 1].c_str(), ms);
        }
    }

    size_t getNumRecords() const{
		return names_.size();
    }

    /**
     * reset all measurments
     */
    void reset(){
        names_.clear();
    }

    /**
     * read out measurements
     */
    float getScopeTime(int start = - 1, int end = -1) const{
        if(end < 0)
            end = names_.size()-1;
        if(start < 0)
            start = 0;
		if (end >= names_.size() || start >= names_.size())
            return -1.0f;

        cudaEventSynchronize(events_[end]);

        float ms;
		cudaEventElapsedTime(&ms, events_[start], events_[end]);
        return ms;
    }

	// return the current time 
	float getCurrentTime(){
		if (names_.size() < 2)
			return -1.0f;
		cudaEventSynchronize(events_[names_.size() - 1]);

		float ms;
		cudaEventElapsedTime(&ms, events_[names_.size() - 2], events_[names_.size()-1]);
		return ms;
	}

private:
    std::vector<cudaEvent_t> events_; // the CUDA events used to measure timings
    std::vector<std::string> names_; // optional names provided for the individual events
};

#endif /* EVENTMANAGEMENT_CUH_ */
