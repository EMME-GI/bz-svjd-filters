#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "kfdnf_core.cpp" // Must be the name of the C++ file!

namespace py = pybind11;

PYBIND11_MODULE(kfdnf_fast, m) {
    m.doc() = "C++ implementation of SVJD filters (KF-DNF, FO, PF)";

    // 2. Feunou & Okou (FO)
    py::class_<FeunouOkouKF>(m, "FeunouOkouKF")
        .def(py::init<>())
        .def("filter_series", &FeunouOkouKF::filter_series)
        .def("get_v_history", &FeunouOkouKF::get_v_history);

    // 3. Discrete Nonlinear Filter (DNF)
    py::class_<DiscreteNonlinearFilter>(m, "DiscreteNonlinearFilter")
        .def(py::init<>())
        .def("filter_series", &DiscreteNonlinearFilter::filter_series)
        .def("get_v_history", &DiscreteNonlinearFilter::get_v_history);

    // 4. Particle Filter (PF)
    py::class_<ParticleFilter>(m, "ParticleFilter")
        .def(py::init<>())
        // Note: the Particle Filter in our C++ code have no init_state
        .def("filter_series", &ParticleFilter::filter_series)
        .def("get_v_history", &ParticleFilter::get_v_history);
}