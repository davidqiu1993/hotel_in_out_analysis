#!/usr/bin/env python

"""
launch_analysis.py
Launch the analysis of hotel checkin and checkout activities.
"""

__version__     = "0.0.1"
__author__      = "David Qiu"
__email__       = "dq@cs.cmu.edu"
__website__     = "http://www.davidqiu.com/"
__copyright__   = "Copyright (C) 2018, David Qiu. All rights reserved."

import csv
import datetime
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

import IPython


def load_dataset(verbose=False):
  # load dataset
  with open('data/Bookings_dump-Table 1.csv', 'r') as csvfile:
    fields = []
    dataset = []

    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
      if line_count == 0:
        fields = row
        if verbose:
          print(f'Column names are {", ".join(row)}')
      else:
        dataset.append(row)

      line_count += 1

    if verbose:
      print('Processed %s lines.' % (len(dataset)))

  # construct date time dataset
  dts = []
  for i in range(len(dataset)):
    d = dataset[i]

    # check if data sample is complete and valid
    valid = True
    for j in [5, 6, 7, 8, 9, 10, 11]:
      if d[j] == 'NULL':
        valid = False
        break
    if not valid:
      continue

    # parse booking date time
    dt_booking = datetime.datetime.strptime(d[5] + ' ' + d[6], '%m/%d/%y %H:%M:%S')

    # parse guest checkin date time
    dt_guest_checkin = datetime.datetime.strptime(d[7] + ' 12:00:00', '%m/%d/%y %H:%M:%S')

    # parse pm checkin date time
    dt_pm_checkin = datetime.datetime.strptime(d[8] + ' ' + d[9], '%m/%d/%y %H:%M:%S')

    # parse guest checkout date time
    dt_guest_checkout = datetime.datetime.strptime(d[7] + ' 12:00:00', '%m/%d/%y %H:%M:%S') + datetime.timedelta(int(d[13]), 0)

    # parse pm checkout date time
    dt_pm_checkout = datetime.datetime.strptime(d[10] + ' ' + d[11], '%m/%d/%y %H:%M:%S')

    # add to list
    dts.append((dt_booking, dt_guest_checkin, dt_pm_checkin, dt_guest_checkout, dt_pm_checkout))

  # convert dataset format
  # - [0] booking: date, time (not quite relevant)
  # - [1] guest_checkin: date (since no one knows when the guest came exactly)
  # - [2] pm_checkin: date, time
  # - [3] guest_checkout: date (since no one knows when the guest left exactly)
  # - [4] pm_checkout: date, time
  T = np.array(dts)

  return T


def calculate_checkin_hours_samples(T, range_checkin_hours):
  vec_checkin_hours = T[:,2] - T[:,1] + datetime.timedelta(0, 3600 * 12)
  checkin_hours = []
  for i in range(len(vec_checkin_hours)):
    sample = vec_checkin_hours[i].total_seconds() / (3600)
    if 0 <= sample and sample < range_checkin_hours:
      checkin_hours.append(sample)

  return checkin_hours


def calculate_checkout_hours_samples(T, range_checkout_hours):
  vec_checkout_hours = T[:,4] - T[:,3] + datetime.timedelta(0, 3600 * 12)
  checkout_hours = []
  for i in range(len(vec_checkout_hours)):
    sample = vec_checkout_hours[i].total_seconds() / (3600)
    if 0 <= sample and sample < range_checkout_hours:
      checkout_hours.append(sample)

  return checkout_hours


def analyze_checkin_hours(T, draw=False, density=False, alpha=1.0, verbose=False):
  RANGE_CHECKIN_HOURS = 48
  RESOLUTION_CHECKIN = 0.5

  range_checkin_slots = RANGE_CHECKIN_HOURS / RESOLUTION_CHECKIN

  # calculate checkin hours samples
  checkin_hours = calculate_checkin_hours_samples(T, RANGE_CHECKIN_HOURS)

  if verbose:
    print('len(checkin_hours) = %d' % (len(checkin_hours)))

  n, bins, patches = (None, None, None)
  if draw:
    n, bins, patches = plt.hist(checkin_hours, 
      range=(0, RANGE_CHECKIN_HOURS), 
      bins=np.linspace(0, range_checkin_slots, range_checkin_slots + 1) * RESOLUTION_CHECKIN,
      edgecolor='black',
      linewidth=1,
      density=density,
      alpha=alpha
    )

    plt.axvline(0, color='black', linestyle='dashed', linewidth=1)
    plt.axvline(12, color='black', linestyle='dashed', linewidth=1)
    plt.axvline(24, color='black', linestyle='dashed', linewidth=1)
    plt.axvline(36, color='black', linestyle='dashed', linewidth=1)
    plt.axvline(48, color='black', linestyle='dashed', linewidth=1)

    plt.plot()

  return checkin_hours, n, bins, patches


def analyze_checkout_hours(T, draw=False, density=False, alpha=1.0, verbose=False):
  RANGE_CHECKOUT_HOURS = 48
  RESOLUTION_CHECKOUT = 0.5

  range_checkout_slots = RANGE_CHECKOUT_HOURS / RESOLUTION_CHECKOUT

  # calculate checkout hours samples
  checkout_hours = calculate_checkout_hours_samples(T, RANGE_CHECKOUT_HOURS)

  if verbose:
    print('len(checkout_hours) = %d' % (len(checkout_hours)))

  n, bins, patches = (None, None, None)
  if draw:
    n, bins, patches = plt.hist(checkout_hours, 
      range=(0, RANGE_CHECKOUT_HOURS), 
      bins=np.linspace(0, range_checkout_slots, range_checkout_slots + 1) * RESOLUTION_CHECKOUT,
      edgecolor='black',
      linewidth=1,
      density=density,
      alpha=alpha
    )

    plt.axvline(0, color='black', linestyle='dashed', linewidth=1)
    plt.axvline(12, color='black', linestyle='dashed', linewidth=1)
    plt.axvline(24, color='black', linestyle='dashed', linewidth=1)
    plt.axvline(36, color='black', linestyle='dashed', linewidth=1)
    plt.axvline(48, color='black', linestyle='dashed', linewidth=1)

    plt.plot()

  return checkout_hours, n, bins, patches


def analyze_cross_events(T, draw=False, verbose=False):
  checkin_hours, checkin_n, checkin_bins, checkin_patches = analyze_checkin_hours(T, draw=True, density=True, alpha=0.5, verbose=False)
  checkout_hours, checkout_n, checkout_bins, checkout_patches = analyze_checkout_hours(T, draw=True, density=True, alpha=0.5, verbose=False)

  checkin_freq = checkin_n / np.sum(checkin_n)
  checkout_freq = checkout_n / np.sum(checkout_n)

  if draw:
    #plt.plot(np.linspace(0.5, len(checkin_freq) - 0.5, len(checkin_freq)), checkin_freq)
    #plt.plot(np.linspace(0.5, len(checkout_freq) - 0.5, len(checkout_freq)), checkout_freq)
    pass

  return checkin_freq, checkout_freq


def model_checkin_hours(T, draw=False, verbose=False):
  RANGE_DISP = 48
  RANGE_MODEL = (8, 30)
  RESOLUTION = 0.5

  samples_disp = calculate_checkin_hours_samples(T, RANGE_DISP)
  samples_model = []
  for s in samples_disp:
    if RANGE_MODEL[0] < s and s < RANGE_MODEL[1]:
      samples_model.append(s)

  if verbose:
    print('len(samples_model) = %d' % (len(samples_model)))

  mu = [ 15, 22 ]
  sigma = [ 1, 1 ]
  datasets = [ [], [] ]

  if verbose:
    print('iter #%d: mu=[%.2f, %.2f], sigma=[%.2f, %.2f], err=max' % (
      0, mu[0], mu[1], sigma[0], sigma[1]))

  err = 1
  epsilon = 1e-6
  iter_count = 0
  while err > epsilon:
    mu0 = np.array(mu)
    sigma0 = np.array(sigma)

    iter_count = iter_count + 1

    # E-step
    datasets[0] = []
    datasets[1] = []
    
    P = [ [], [] ]
    P[0] = stats.norm.pdf(samples_model, mu[0], sigma[0])
    P[1] = stats.norm.pdf(samples_model, mu[1], sigma[1])

    for i in range(len(samples_model)):
      if P[0][i] > P[1][i]:
        datasets[0].append(samples_model[i])
      else:
        datasets[1].append(samples_model[i])

    # M-step
    for i in [0, 1]:
      mu[i] = np.mean(datasets[i])
      sigma[i] = np.std(datasets[i])
    mu = np.array(mu)
    sigma = np.array(sigma)

    # update error
    err = np.sum(np.abs(mu - mu0)) + np.sum(np.abs(sigma - sigma0))

    if verbose:
      print('iter #%d: mu=[%.2f, %.2f], sigma=[%.2f, %.2f], err=%.6f' % (
        iter_count, mu[0], mu[1], sigma[0], sigma[1], err))

  if verbose:
    print('mu=[%f, %f]' % (mu[0], mu[1]))
    print('sigma=[%f, %f]' % (sigma[0] ** 2, sigma[1] ** 2))
    print('err=%f' % (err))

  if draw:
    analyze_checkin_hours(T, draw=True, density=True, alpha=0.5, verbose=False)

    slots = RANGE_DISP / RESOLUTION
    X = np.linspace(0, slots, slots + 1) * RESOLUTION
    Y = (stats.norm.pdf(X, mu[0], sigma[0]) + stats.norm.pdf(X, mu[1], sigma[1])) / 2
    plt.plot(X, Y, '-o')

  return mu, sigma


def model_checkout_hours(T, draw=False, verbose=False):
  RANGE_DISP = 48
  RANGE_MODEL = (3, 22)
  RESOLUTION = 0.5

  samples_disp = calculate_checkout_hours_samples(T, RANGE_DISP)
  samples_model = []
  for s in samples_disp:
    if RANGE_MODEL[0] < s and s < RANGE_MODEL[1]:
      samples_model.append(s)

  if verbose:
    print('len(samples_model) = %d' % (len(samples_model)))

  mu_G = np.mean(samples_model)
  sigma_G = np.std(samples_model)

  if verbose:
    print('mu_G = %f' % (mu_G))
    print('sigma_G^2 = %f' % (sigma_G ** 2))

  if draw:
    analyze_checkout_hours(T, draw=True, density=True, alpha=0.5, verbose=False)

    slots = RANGE_DISP / RESOLUTION
    X = np.linspace(0, slots, slots + 1) * RESOLUTION
    Y = stats.norm.pdf(X, mu_G, sigma_G)
    plt.plot(X, Y, '-o')

  return mu_G, sigma_G


def main():
  # load dataset
  # - [0] booking: date, time (not quite relevant)
  # - [1] guest_checkin: date (since no one knows when the guest came exactly)
  # - [2] pm_checkin: date, time
  # - [3] guest_checkout: date (since no one knows when the guest left exactly)
  # - [4] pm_checkout: date, time
  T = load_dataset(verbose=True)
  print('')

  # analyze checkin hours
  if False:
    print('analysis: checkin hours')
    analyze_checkin_hours(T, draw=True, verbose=True)
    print('')

  # analyze checkout hours
  if False:
    print('analysis: checkout hours')
    analyze_checkout_hours(T, draw=True, verbose=True)
    print('')

  # analyze cross events
  if False:
    print('analysis: cross events')
    analyze_cross_events(T, draw=True, verbose=True)
    print('')

  # model checkin hours
  if True:
    print('model: checkin hours')
    model_checkin_hours(T, draw=True, verbose=True)
    print('')

  # model checkout hours
  if False:
    print('model: checkout hours')
    model_checkout_hours(T, draw=True, verbose=True)
    print('')

  # show drawings
  plt.show()

  #IPython.embed()


if __name__ == '__main__':
  main()
