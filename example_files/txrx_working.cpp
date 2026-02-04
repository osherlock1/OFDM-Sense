
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

namespace po = boost::program_options;

/***********************************************************************
 * Signal handlers
 **********************************************************************/
static bool stop_signal_called = false;
void sig_int_handler(int) { stop_signal_called = true; }

/***********************************************************************
 * Utilities
 **********************************************************************/
// Change to filename, e.g. from usrp_samples.dat to usrp_samples.00.dat
// but only if multiple names are to be generated.
static std::string generate_out_filename(
    const std::string& base_fn, size_t n_names, size_t this_name)
{
    if (n_names == 1) {
        return base_fn;
    }

    boost::filesystem::path base_fn_fp(base_fn);
    base_fn_fp.replace_extension(boost::filesystem::path(
        str(boost::format("%02d%s") % this_name % base_fn_fp.extension().string())));
    return base_fn_fp.string();
}

/***********************************************************************
 * TX worker (generated waveform)
 **********************************************************************/
static void transmit_worker_wave(std::vector<std::complex<float>> buff,
    wave_table_class wave_table,
    uhd::tx_streamer::sptr tx_streamer,
    uhd::tx_metadata_t metadata,
    size_t step,
    size_t index,
    int num_channels)
{
    std::vector<std::complex<float>*> buffs(num_channels, &buff.front());

    while (not stop_signal_called) {
        for (size_t n = 0; n < buff.size(); n++) {
            buff[n] = wave_table(index += step);
        }
        tx_streamer->send(buffs, buff.size(), metadata);
        metadata.start_of_burst = false;
        metadata.has_time_spec  = false;
    }
    metadata.end_of_burst = true;
    tx_streamer->send("", 0, metadata);
}

/***********************************************************************
 * TX worker (read samples from file, time-scheduled start)
 **********************************************************************/
template <typename samp_type>
static void transmit_worker_file(uhd::tx_streamer::sptr tx_stream,
    const std::string& file,
    size_t samps_per_buff,
    uhd::time_spec_t t0,
    bool repeat,
    int num_channels)
{
    do {
        std::ifstream infile(file.c_str(), std::ifstream::binary);
        if (!infile.is_open()) {
            throw std::runtime_error("Cannot open --tx-file: " + file);
        }

        uhd::tx_metadata_t md;
        md.start_of_burst = true;
        md.end_of_burst   = false;
        md.has_time_spec  = true;  // schedule aligned start
        md.time_spec      = t0;

        std::vector<samp_type>  buff(samps_per_buff);
        std::vector<samp_type*> buffs(tx_stream->get_num_channels(), &buff.front());

        bool first_packet = true;
        while (not stop_signal_called) {
            infile.read(reinterpret_cast<char*>(&buff.front()),
                buff.size() * sizeof(samp_type));
            const size_t num_tx_samps = size_t(infile.gcount() / sizeof(samp_type));

            if (num_tx_samps == 0) {
                // EOF: send EOB and stop this iteration
                md.end_of_burst = true;
                tx_stream->send("", 0, md);
                break;
            }

            const size_t sent = tx_stream->send(buffs, num_tx_samps, md);
            if (sent != num_tx_samps) {
                std::cerr << "TX stream timeout: requested " << num_tx_samps
                          << ", actually sent " << sent << " samples.\n";
                break;
            }

            if (first_packet) {
                md.start_of_burst = false;
                md.has_time_spec  = false;
                first_packet      = false;
            }
        }

        infile.close();
        // For repeats, subsequent passes start immediately (no rescheduling)
        t0 = uhd::time_spec_t(0.0);
        if (repeat && !stop_signal_called) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    } while (repeat && !stop_signal_called);
}

/***********************************************************************
 * recv_to_file function (unchanged from example; RX aligns to TX t0)
 **********************************************************************/
template <typename samp_type>
static void recv_to_file(uhd::usrp::multi_usrp::sptr usrp,
    const std::string& cpu_format,
    const std::string& wire_format,
    const std::string& file,
    size_t samps_per_buff,
    int num_requested_samples,
    uhd::time_spec_t start_time,
    double settling_time,
    std::vector<size_t> rx_channel_nums)
{
    int num_total_samps = 0;

    uhd::stream_args_t stream_args(cpu_format, wire_format);
    stream_args.channels             = rx_channel_nums;
    uhd::rx_streamer::sptr rx_stream = usrp->get_rx_stream(stream_args);

    uhd::rx_metadata_t md;
    std::vector<std::vector<samp_type>> buffs(
        rx_channel_nums.size(), std::vector<samp_type>(samps_per_buff));
    std::vector<samp_type*> buff_ptrs;
    for (size_t i = 0; i < buffs.size(); i++) buff_ptrs.push_back(&buffs[i].front());

    std::vector<std::shared_ptr<std::ofstream>> outfiles;
    for (size_t i = 0; i < buffs.size(); i++) {
        const std::string this_filename = generate_out_filename(file, buffs.size(), i);
        outfiles.push_back(std::shared_ptr<std::ofstream>(
            new std::ofstream(this_filename.c_str(), std::ofstream::binary)));
    }

    bool overflow_message = true;
    double timeout        = settling_time + 0.5; // generous first timeout

    uhd::stream_cmd_t stream_cmd((num_requested_samples == 0)
                                     ? uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS
                                     : uhd::stream_cmd_t::STREAM_MODE_NUM_SAMPS_AND_DONE);
    stream_cmd.num_samps  = num_requested_samples;
    stream_cmd.stream_now = false;
    stream_cmd.time_spec  = start_time + uhd::time_spec_t(settling_time);
    rx_stream->issue_stream_cmd(stream_cmd);

    while (not stop_signal_called
           && (num_requested_samples > num_total_samps || num_requested_samples == 0)) {
        size_t num_rx_samps = rx_stream->recv(buff_ptrs, samps_per_buff, md, timeout);
        timeout             = 0.1; // small timeout for subsequent recv

        if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_TIMEOUT) {
            std::cout << "Timeout while streaming" << std::endl;
            break;
        }
        if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_OVERFLOW) {
            if (overflow_message) {
                overflow_message = false;
                std::cerr << boost::format(
                                 "Got an overflow indication. Please consider the following:\n"
                                 "  Your write medium must sustain a rate of %fMB/s.\n"
                                 "  Dropped samples will not be written to the file.\n"
                                 "  Please modify this example for your purposes.\n"
                                 "  This message will not appear again.\n")
                                     % (usrp->get_rx_rate() * sizeof(samp_type) / 1e6);
            }
            continue;
        }
        if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE) {
            throw std::runtime_error("Receiver error " + md.strerror());
        }

        num_total_samps += static_cast<int>(num_rx_samps);
        for (size_t i = 0; i < outfiles.size(); i++) {
            outfiles[i]->write((const char*)buff_ptrs[i],
                static_cast<std::streamsize>(num_rx_samps * sizeof(samp_type)));
        }
    }

    stream_cmd.stream_mode = uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS;
    rx_stream->issue_stream_cmd(stream_cmd);
    for (size_t i = 0; i < outfiles.size(); i++) outfiles[i]->close();
}

/***********************************************************************
 * Main
 **********************************************************************/
int UHD_SAFE_MAIN(int argc, char* argv[])
{
    // TX-side
    std::string tx_args, wave_type, tx_ant, tx_subdev, ref, otw, tx_channels;
    double tx_rate, tx_freq, tx_gain, wave_freq, tx_bw;
    float ampl;

    // RX-side
    std::string rx_args, file, type, rx_ant, rx_subdev, rx_channels;
    size_t total_num_samps, spb;
    double rx_rate, rx_freq, rx_gain, rx_bw;
    double settling;

    // NEW: TX-from-file options
    std::string tx_file, tx_type;
    size_t tx_spb = 0;
    bool tx_repeat = false;

    // Options
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "help message")
        ("tx-args", po::value<std::string>(&tx_args)->default_value(""), "uhd transmit device address args")
        ("rx-args", po::value<std::string>(&rx_args)->default_value(""), "uhd receive device address args")
        ("file", po::value<std::string>(&file)->default_value("usrp_samples.dat"), "name of the file to write binary samples to")
        ("type", po::value<std::string>(&type)->default_value("short"), "RX sample type in file: double, float, or short")
        ("nsamps", po::value<size_t>(&total_num_samps)->default_value(0), "total number of samples to receive")
        ("settling", po::value<double>(&settling)->default_value(double(0.2)), "settling time (seconds) before receiving")
        ("spb", po::value<size_t>(&spb)->default_value(0), "samples per buffer (RX or waveform TX), 0 for default")
        ("tx-rate", po::value<double>(&tx_rate), "rate of transmit outgoing samples")
        ("rx-rate", po::value<double>(&rx_rate), "rate of receive incoming samples")
        ("tx-freq", po::value<double>(&tx_freq), "transmit RF center frequency in Hz")
        ("rx-freq", po::value<double>(&rx_freq), "receive RF center frequency in Hz")
        ("ampl", po::value<float>(&ampl)->default_value(float(0.3)), "amplitude of the generated waveform [0 to 0.7]")
        ("tx-gain", po::value<double>(&tx_gain), "gain for the transmit RF chain")
        ("rx-gain", po::value<double>(&rx_gain), "gain for the receive RF chain")
        ("tx-ant", po::value<std::string>(&tx_ant), "transmit antenna selection")
        ("rx-ant", po::value<std::string>(&rx_ant), "receive antenna selection")
        ("tx-subdev", po::value<std::string>(&tx_subdev), "transmit subdevice specification")
        ("rx-subdev", po::value<std::string>(&rx_subdev), "receive subdevice specification")
        ("tx-bw", po::value<double>(&tx_bw), "analog transmit filter bandwidth in Hz")
        ("rx-bw", po::value<double>(&rx_bw), "analog receive filter bandwidth in Hz")
        ("wave-type", po::value<std::string>(&wave_type)->default_value("CONST"), "waveform type (CONST, SQUARE, RAMP, SINE)")
        ("wave-freq", po::value<double>(&wave_freq)->default_value(0), "waveform frequency in Hz")
        ("ref", po::value<std::string>(&ref), "reference source (internal, external, gpsdo, mimo)")
        ("otw", po::value<std::string>(&otw)->default_value("sc16"), "specify the over-the-wire sample mode")
        ("tx-channels", po::value<std::string>(&tx_channels)->default_value("0"), "which TX channel(s) to use (\"0\", \"1\", \"0,1\", etc)")
        ("rx-channels", po::value<std::string>(&rx_channels)->default_value("0"), "which RX channel(s) to use (\"0\", \"1\", \"0,1\", etc)")
        ("tx-int-n", "tune USRP TX with integer-N tuning")
        ("rx-int-n", "tune USRP RX with integer-N tuning")
        // NEW
        ("tx-file", po::value<std::string>(&tx_file)->default_value(""), "file with TX IQ samples (enables TX-from-file mode)")
        ("tx-type", po::value<std::string>(&tx_type)->default_value("short"), "TX file sample type: double, float, or short")
        ("tx-spb", po::value<size_t>(&tx_spb)->default_value(0), "TX samples-per-buffer for file mode, 0 => auto")
        ("tx-repeat", po::bool_switch(&tx_repeat)->default_value(false), "repeat TX file until Ctrl-C")
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << "UHD TXRX Loopback to File (with TX-from-file)\n" << desc << std::endl;
        return ~0;
    }

    // Create device (use rx_args if provided, else tx_args)
    const std::string dev_args = rx_args.empty() ? tx_args : rx_args;
    std::cout << "\nCreating USRP device with: " << dev_args << "...\n";
    uhd::usrp::multi_usrp::sptr usrp = uhd::usrp::multi_usrp::make(dev_args);

    // Subdevs (default to A:0 if unspecified)
    if (vm.count("tx-subdev")) usrp->set_tx_subdev_spec(uhd::usrp::subdev_spec_t(tx_subdev));
    else                        usrp->set_tx_subdev_spec(uhd::usrp::subdev_spec_t("A:0"));

    if (vm.count("rx-subdev")) usrp->set_rx_subdev_spec(uhd::usrp::subdev_spec_t(rx_subdev));
    else                        usrp->set_rx_subdev_spec(uhd::usrp::subdev_spec_t("A:0"));

    // Channels
    std::vector<std::string> tx_channel_strings;
    std::vector<size_t> tx_channel_nums;
    boost::split(tx_channel_strings, tx_channels, boost::is_any_of("\"',"));
    for (const auto& s : tx_channel_strings) {
        const size_t ch = std::stoul(s);
        if (ch >= usrp->get_tx_num_channels()) throw std::runtime_error("Invalid TX channel");
        tx_channel_nums.push_back(ch);
    }
    std::vector<std::string> rx_channel_strings;
    std::vector<size_t> rx_channel_nums;
    boost::split(rx_channel_strings, rx_channels, boost::is_any_of("\"',"));
    for (const auto& s : rx_channel_strings) {
        const size_t ch = std::stoul(s);
        if (ch >= usrp->get_rx_num_channels()) throw std::runtime_error("Invalid RX channel");
        rx_channel_nums.push_back(ch);
    }

    // Clock source (optional)
    if (vm.count("ref")) {
        usrp->set_clock_source(ref);
    }

    std::cout << "Using Device: " << usrp->get_pp_string() << std::endl;

    // Rates
    if (not vm.count("tx-rate")) { std::cerr << "Specify --tx-rate\n"; return ~0; }
    std::cout << boost::format("Setting TX Rate: %f Msps...") % (tx_rate / 1e6) << std::endl;
    usrp->set_tx_rate(tx_rate);  std::cout << boost::format("Actual TX Rate: %f Msps...\n") % (usrp->get_tx_rate()/1e6) << std::endl;

    if (not vm.count("rx-rate")) { std::cerr << "Specify --rx-rate\n"; return ~0; }
    std::cout << boost::format("Setting RX Rate: %f Msps...") % (rx_rate / 1e6) << std::endl;
    usrp->set_rx_rate(rx_rate);  std::cout << boost::format("Actual RX Rate: %f Msps...\n") % (usrp->get_rx_rate()/1e6) << std::endl;

    // TX tuning/gain/BW
    if (not vm.count("tx-freq")) { std::cerr << "Specify --tx-freq\n"; return ~0; }
    for (size_t ch : tx_channel_nums) {
        std::cout << boost::format("Setting TX Freq: %f MHz...") % (tx_freq / 1e6) << std::endl;
        uhd::tune_request_t tx_tune_request(tx_freq);
        if (vm.count("tx-int-n")) tx_tune_request.args = uhd::device_addr_t("mode_n=integer");
        usrp->set_tx_freq(tx_tune_request, ch);
        std::cout << boost::format("Actual TX Freq: %f MHz...\n") % (usrp->get_tx_freq(ch)/1e6) << std::endl;
        if (vm.count("tx-gain")) { usrp->set_tx_gain(tx_gain, ch); std::cout << "TX Gain: " << usrp->get_tx_gain(ch) << " dB\n"; }
        if (vm.count("tx-bw"))   { usrp->set_tx_bandwidth(tx_bw, ch); std::cout << "TX BW: " << usrp->get_tx_bandwidth(ch) << " Hz\n"; }
        if (vm.count("tx-ant"))  { usrp->set_tx_antenna(tx_ant, ch); }
    }

    // RX tuning/gain/BW
    if (not vm.count("rx-freq")) { std::cerr << "Specify --rx-freq\n"; return ~0; }
    for (size_t ch : rx_channel_nums) {
        std::cout << boost::format("Setting RX Freq: %f MHz...") % (rx_freq / 1e6) << std::endl;
        uhd::tune_request_t rx_tune_request(rx_freq);
        if (vm.count("rx-int-n")) rx_tune_request.args = uhd::device_addr_t("mode_n=integer");
        usrp->set_rx_freq(rx_tune_request, ch);
        std::cout << boost::format("Actual RX Freq: %f MHz...\n") % (usrp->get_rx_freq(ch)/1e6) << std::endl;
        if (vm.count("rx-gain")) { usrp->set_rx_gain(rx_gain, ch); std::cout << "RX Gain: " << usrp->get_rx_gain(ch) << " dB\n"; }
        if (vm.count("rx-bw"))   { usrp->set_rx_bandwidth(rx_bw, ch); std::cout << "RX BW: " << usrp->get_rx_bandwidth(ch) << " Hz\n"; }
        if (vm.count("rx-ant"))  { usrp->set_rx_antenna(rx_ant, ch); }
    }

    // Waveform sanity if not using tx-file
    if (tx_file.empty()) {
        if (wave_freq == 0 && wave_type == "CONST") wave_freq = usrp->get_tx_rate() / 2;
        if (std::abs(wave_freq) > usrp->get_tx_rate() / 2)
            throw std::runtime_error("wave freq out of Nyquist zone");
        if (usrp->get_tx_rate() / std::abs(wave_freq) > wave_table_len / 2)
            throw std::runtime_error("wave freq too small for table");
    }

    // Precompute waveform (only used if tx_file empty)
    const wave_table_class wave_table(wave_type, ampl);
    const size_t step = tx_file.empty() ? std::lround(wave_freq / usrp->get_tx_rate() * wave_table_len) : 0;
    size_t index      = 0;

    // Create TX streamer with CPU format depending on mode
    std::string tx_cpu_fmt = "fc32"; // default (generated float32)
    if (!tx_file.empty()) {
        if (tx_type == "double")      tx_cpu_fmt = "fc64";
        else if (tx_type == "float")  tx_cpu_fmt = "fc32";
        else if (tx_type == "short")  tx_cpu_fmt = "sc16";
        else throw std::runtime_error("Unknown --tx-type " + tx_type);
    }
    uhd::stream_args_t tx_sa(tx_cpu_fmt, otw);
    tx_sa.channels = tx_channel_nums;
    uhd::tx_streamer::sptr tx_stream = usrp->get_tx_stream(tx_sa);

    // Allocate TX buffer sizes
    if (tx_spb == 0) tx_spb = spb;                      // prefer explicit tx-spb if given
    if (tx_spb == 0) tx_spb = tx_stream->get_max_num_samps() * 10; // auto
    std::vector<std::complex<float>> buff(tx_spb); // only used in waveform mode
    const int num_tx_channels = static_cast<int>(tx_channel_nums.size());

    // Optional LO lock checks (if available)
    auto tx_sensor_names = usrp->get_tx_sensor_names(0);
    if (std::find(tx_sensor_names.begin(), tx_sensor_names.end(), "lo_locked") != tx_sensor_names.end()) {
        auto lo_locked = usrp->get_tx_sensor("lo_locked", 0);
        std::cout << boost::format("Checking TX: %s ...") % lo_locked.to_pp_string() << std::endl;
        UHD_ASSERT_THROW(lo_locked.to_bool());
    }
    auto rx_sensor_names = usrp->get_rx_sensor_names(0);
    if (std::find(rx_sensor_names.begin(), rx_sensor_names.end(), "lo_locked") != rx_sensor_names.end()) {
        auto lo_locked = usrp->get_rx_sensor("lo_locked", 0);
        std::cout << boost::format("Checking RX: %s ...") % lo_locked.to_pp_string() << std::endl;
        UHD_ASSERT_THROW(lo_locked.to_bool());
    }

    if (total_num_samps == 0) {
        std::signal(SIGINT, &sig_int_handler);
        std::cout << "Press Ctrl + C to stop streaming..." << std::endl;
    }

    // Reset time and schedule aligned TX/RX
    std::cout << boost::format("Setting device timestamp to 0...") << std::endl;
    usrp->set_time_now(uhd::time_spec_t(0.0));
    const auto t0 = usrp->get_time_now() + uhd::time_spec_t(0.5);

    // TX metadata seed (used only by waveform worker)
    uhd::tx_metadata_t md;
    md.start_of_burst = true;
    md.end_of_burst   = false;
    md.has_time_spec  = true;
    md.time_spec      = t0; // give time to fill TX buffers

    // Launch TX worker (file or waveform)
    std::thread tx_thread;
    if (!tx_file.empty()) {
        if (tx_type == "double")
            tx_thread = std::thread(transmit_worker_file<std::complex<double>>, tx_stream, tx_file, tx_spb, t0, tx_repeat, num_tx_channels);
        else if (tx_type == "float")
            tx_thread = std::thread(transmit_worker_file<std::complex<float>>,  tx_stream, tx_file, tx_spb, t0, tx_repeat, num_tx_channels);
        else if (tx_type == "short")
            tx_thread = std::thread(transmit_worker_file<std::complex<short>>,  tx_stream, tx_file, tx_spb, t0, tx_repeat, num_tx_channels);
        else throw std::runtime_error("Unknown --tx-type " + tx_type);
    } else {
        tx_thread = std::thread(transmit_worker_wave, buff, wave_table, tx_stream, md, step, index, num_tx_channels);
    }

    // RX to file (aligned start time)
    if (type == "double") {
        recv_to_file<std::complex<double>>(usrp, "fc64", otw, file, (spb?spb:tx_spb),
            static_cast<int>(total_num_samps), t0, settling, rx_channel_nums);
    } else if (type == "float") {
        recv_to_file<std::complex<float>>(usrp, "fc32", otw, file, (spb?spb:tx_spb),
            static_cast<int>(total_num_samps), t0, settling, rx_channel_nums);
    } else if (type == "short") {
        recv_to_file<std::complex<short>>(usrp, "sc16", otw, file, (spb?spb:tx_spb),
            static_cast<int>(total_num_samps), t0, settling, rx_channel_nums);
    } else {
        stop_signal_called = true;
        tx_thread.join();
        throw std::runtime_error("Unknown RX --type " + type);
    }

    // Clean up TX worker
    stop_signal_called = true;
    tx_thread.join();

    std::cout << "\nDone!\n\n";
    return EXIT_SUCCESS;
}