spec_granger
============

This is a python implementation of the spectral granger computed from the cross spectral densities



format of matrices
==================

Most matrices like S (input/output), H (output), Z(output), S(output) share a common format.

for the spectral density S this is for neurons 1 and 2:
[[S11, S12],[S21, S22]]

where S11 is the auto-spectral density of 1
and S12 is the cross-spectral density of 1 and 2

Many of the matrices also have a frequency dimension, thus their shape is
(2,2,freqs). S(input) is two-sided in terms of the frequency spectrum
thus from -nyquist to nyquist. (you can plug in the result from an fft directly)

Other matrices such as H and S(output) are one-sided. Z has no frequency dimension because it represents the noise covariances.




