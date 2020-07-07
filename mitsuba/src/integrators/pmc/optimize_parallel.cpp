constexpr static int N_THREADS = 128;

void optimize(int emIterations) {
    std::vector<std::thread> threads; 
    for(int i = 0; i < N_THREADS; ++i) {
        threads.push_back(std::thread([=]() {
            int begin = m_samples->size() / 128 * i;
            int end = m_samples->size() / 128 * (i + 1);
            std::cerr << "Thread " << i << ": optimizing from " << begin << " to " << end << std::endl;
            m_optimizer->optimize(*m_distribution, *m_samples, emIterations, i, begin, end);
        }));
    }
    for(auto& thread : threads) {
        if(thread.joinable()) {
            thread.join();
        }
    }
}

