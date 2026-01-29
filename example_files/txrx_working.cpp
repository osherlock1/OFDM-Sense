#include "wavetable.hpp"
#include <uhd/exception.hpp>
#include <uhd/types/tune_request.hpp>
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/utils/safe_main.hpp>
#include <uhd/utils/static.hpp>
#include <uhd/utils/thread.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <csignal>
#include <fstream>
#include <functional>
#include <iostream>
#include <thread>
#include <vector>

namespace po = boost::program_options;

static bool stop_signal_called = false;
void sig_int_handler(int) { stop_signal_called = true; }

static std::string generate_out_filename(const std::string& base_fn, size_t n_names, size_t this_name) {
    if (n_names == 1) return base_fn;
    boost::filesystem::path base_fn_fp(base_fn);
    base_fn_fp.replace_extension(boost::filesystem::path(str(boost::format("%02d%s") % this_name % base_fn_fp.extension().string())));
    return base_fn_fp.string();
}

/***********************************************************************
 * Worker Functions
 **********************************************************************/
template <typename samp_type>
static void transmit_worker_file(uhd::tx_streamer::sptr tx_stream, const std::string& file, size_t samps_per_buff, uhd::time_spec_t t0, bool repeat) {
    do {
        std::ifstream infile(file.c_str(), std::ifstream::binary);
        if (!infile.is_open()) throw std::runtime_error("Cannot open --tx-file: " + file);
        uhd::tx_metadata_t md;
        md.start_of_burst = true;
        md.has_time_spec  = true;
        md.time_spec      = t0;
        std::vector<samp_type>  buff(samps_per_buff);
        std::vector<samp_type*> buffs(tx_stream->get_num_channels(), &buff.front());
        while (not stop_signal_called) {
            infile.read(reinterpret_cast<char*>(&buff.front()), buff.size() * sizeof(samp_type));
            const size_t num_tx_samps = size_t(infile.gcount() / sizeof(samp_type));
            if (num_tx_samps == 0) {
                md.end_of_burst = true;
                tx_stream->send("", 0, md);
                break;
            }
            const size_t sent = tx_stream->send(buffs, num_tx_samps, md, 6.0);
            if (sent != num_tx_samps) break;
            md.start_of_burst = false;
            md.has_time_spec  = false;
        }
        infile.close();
        t0 = uhd::time_spec_t(0.0);
    } while (repeat && !stop_signal_called);
}

template <typename samp_type>
static void recv_to_file(uhd::usrp::multi_usrp::sptr usrp, const std::string& cpu_format, const std::string& wire_format, const std::string& file, size_t samps_per_buff, int num_requested_samples, uhd::time_spec_t start_time, double settling_time, std::vector<size_t> rx_channel_nums) {
    int num_total_samps = 0;
    uhd::stream_args_t stream_args(cpu_format, wire_format);
    stream_args.channels = rx_channel_nums;
    uhd::rx_streamer::sptr rx_stream = usrp->get_rx_stream(stream_args);
    uhd::rx_metadata_t md;
    std::vector<std::vector<samp_type>> buffs(rx_channel_nums.size(), std::vector<samp_type>(samps_per_buff));
    std::vector<samp_type*> buff_ptrs;
    for (size_t i = 0; i < buffs.size(); i++) buff_ptrs.push_back(&buffs[i].front());
    std::vector<std::shared_ptr<std::ofstream>> outfiles;
    for (size_t i = 0; i < buffs.size(); i++) {
        outfiles.push_back(std::make_shared<std::ofstream>(generate_out_filename(file, buffs.size(), i), std::ofstream::binary));
    }
    uhd::stream_cmd_t stream_cmd((num_requested_samples == 0) ? uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS : uhd::stream_cmd_t::STREAM_MODE_NUM_SAMPS_AND_DONE);
    stream_cmd.num_samps = num_requested_samples;
    stream_cmd.stream_now = false;
    stream_cmd.time_spec = start_time + uhd::time_spec_t(settling_time);
    rx_stream->issue_stream_cmd(stream_cmd);
    double timeout = settling_time + 1.0;
    while (not stop_signal_called && (num_requested_samples > num_total_samps || num_requested_samples == 0)) {
        size_t num_rx_samps = rx_stream->recv(buff_ptrs, samps_per_buff, md, timeout);
        timeout = 0.5;
        if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_TIMEOUT) { std::cout << "Timeout while streaming" << std::endl; break; }
        if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE && md.error_code != uhd::rx_metadata_t::ERROR_CODE_OVERFLOW) throw std::runtime_error("Receiver error " + md.strerror());
        num_total_samps += static_cast<int>(num_rx_samps);
        for (size_t i = 0; i < outfiles.size(); i++) outfiles[i]->write((const char*)buff_ptrs[i], static_cast<std::streamsize>(num_rx_samps * sizeof(samp_type)));
    }
}

/***********************************************************************
 * Synchronization Helper
 **********************************************************************/
static void synchronize_to_pps(uhd::usrp::multi_usrp::sptr usrp, const std::string& ref_source) {
    if (ref_source == "internal") {
        usrp->set_time_now(uhd::time_spec_t(0.0));
        return;
    }
    std::cout << "\n--- Syncing Multi-USRP to " << ref_source << " (10MHz + PPS) ---" << std::endl;
    usrp->set_clock_source(ref_source);
    usrp->set_time_source(ref_source);

    // Wait for Ref Lock on all motherboards
    for (int i = 0; i < 30; i++) {
        bool all_locked = true;
        for (size_t m = 0; m < usrp->get_num_mboards(); m++) all_locked &= usrp->get_mboard_sensor("ref_locked", m).to_bool();
        if (all_locked) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // Aligned Time Reset: Wait for a PPS edge to pass to avoid race conditions
    uhd::time_spec_t last_pps = usrp->get_time_last_pps();
    while (last_pps == usrp->get_time_last_pps()) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
    
    // Immediately after the pulse, set the time for the NEXT pulse
    usrp->set_time_next_pps(uhd::time_spec_t(0.0));
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));

    // Hardware Verification
    for (size_t m = 0; m < usrp->get_num_mboards(); m++) {
        double diff = std::abs(usrp->get_time_last_pps(m).get_real_secs());
        if (diff > 0.5) throw std::runtime_error("MBoard " + std::to_string(m) + " failed PPS latch!");
    }
    std::cout << "All MBoards Successfully Locked and Time-Synced.\n";
}

/***********************************************************************
 * Main
 **********************************************************************/
int UHD_SAFE_MAIN(int argc, char* argv[]) {
    std::string tx_args, rx_args, file, type, tx_channels, rx_channels, ref, tx_file, tx_type, tx_subdev, rx_subdev, otw;
    double tx_rate, rx_rate, tx_freq, rx_freq, tx_gain, rx_gain, settling;
    size_t total_num_samps, spb, tx_spb;
    bool tx_repeat;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "help message")
        ("tx-args", po::value<std::string>(&tx_args)->default_value(""))
        ("rx-args", po::value<std::string>(&rx_args)->default_value(""))
        ("file", po::value<std::string>(&file)->default_value("usrp_samples.dat"))
        ("type", po::value<std::string>(&type)->default_value("float"))
        ("nsamps", po::value<size_t>(&total_num_samps)->default_value(0))
        ("settling", po::value<double>(&settling)->default_value(0.5))
        ("spb", po::value<size_t>(&spb)->default_value(10000))
        ("tx-rate", po::value<double>(&tx_rate))
        ("rx-rate", po::value<double>(&rx_rate))
        ("tx-freq", po::value<double>(&tx_freq))
        ("rx-freq", po::value<double>(&rx_freq))
        ("tx-gain", po::value<double>(&tx_gain)->default_value(0))
        ("rx-gain", po::value<double>(&rx_gain)->default_value(0))
        ("tx-channels", po::value<std::string>(&tx_channels)->default_value("0"))
        ("rx-channels", po::value<std::string>(&rx_channels)->default_value("0"))
        ("ref", po::value<std::string>(&ref)->default_value("external"))
        ("otw", po::value<std::string>(&otw)->default_value("sc16"))
        ("tx-file", po::value<std::string>(&tx_file)->default_value(""))
        ("tx-type", po::value<std::string>(&tx_type)->default_value("float"))
        ("tx-spb", po::value<size_t>(&tx_spb)->default_value(0)) // FIXED: Matches python script flag
        ("tx-repeat", po::bool_switch(&tx_repeat)->default_value(false))
        ("tx-subdev", po::value<std::string>(&tx_subdev)->default_value("A:0"))
        ("rx-subdev", po::value<std::string>(&rx_subdev)->default_value("A:0"));

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) { std::cout << desc << std::endl; return 0; }

    const std::string dev_args = rx_args.empty() ? tx_args : rx_args;
    auto usrp = uhd::usrp::multi_usrp::make(dev_args);
    
    // RFNoC Fix: Set Master Clock before any other property configuration
    usrp->set_master_clock_rate(200e6);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    // Parse Channel Lists
    std::vector<size_t> tx_chans, rx_chans;
    std::vector<std::string> tx_s, rx_s;
    boost::split(tx_s, tx_channels, boost::is_any_of(","));
    boost::split(rx_s, rx_channels, boost::is_any_of(","));
    for(auto& s : tx_s) tx_chans.push_back(std::stoul(s));
    for(auto& s : rx_s) rx_chans.push_back(std::stoul(s));

    // Configuration
    for (size_t ch : tx_chans) {
        usrp->set_tx_rate(tx_rate, ch);
        usrp->set_tx_subdev_spec(tx_subdev, ch);
    }
    for (size_t ch : rx_chans) {
        usrp->set_rx_rate(rx_rate, ch);
        usrp->set_rx_subdev_spec(rx_subdev, ch);
    }

    // Synchronization
    synchronize_to_pps(usrp, ref);

    // Timed Tuning for Phase Alignment
    uhd::time_spec_t cmd_time = usrp->get_time_now() + uhd::time_spec_t(0.1);
    usrp->set_command_time(cmd_time);
    for (size_t ch : tx_chans) { usrp->set_tx_freq(tx_freq, ch); usrp->set_tx_gain(tx_gain, ch); }
    for (size_t ch : rx_chans) { usrp->set_rx_freq(rx_freq, ch); usrp->set_rx_gain(rx_gain, ch); }
    usrp->clear_command_time();
    std::this_thread::sleep_for(std::chrono::milliseconds(400));

    // Start Time (Scheduled 2 seconds in future to ensure host buffers are ready)
    const auto t0 = usrp->get_time_now() + uhd::time_spec_t(2.0);
    std::signal(SIGINT, &sig_int_handler);

    std::thread tx_thread;
    if (!tx_file.empty()) {
        tx_thread = std::thread([&]() {
            uhd::stream_args_t tx_sa(tx_type == "float" ? "fc32" : "sc16", otw);
            tx_sa.channels = tx_chans;
            auto tx_stream = usrp->get_tx_stream(tx_sa);
            transmit_worker_file<std::complex<float>>(tx_stream, tx_file, (tx_spb ? tx_spb : 10000), t0, tx_repeat);
        });
    }

    recv_to_file<std::complex<float>>(usrp, "fc32", otw, file, spb, total_num_samps, t0, settling, rx_chans);

    stop_signal_called = true;
    if (tx_thread.joinable()) tx_thread.join();

    return EXIT_SUCCESS;
}