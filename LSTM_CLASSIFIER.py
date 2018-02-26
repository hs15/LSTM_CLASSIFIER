"""
LSTM classifier model. Written by Hamid Soleimani
Imperial License
"""
import numpy as np
import array

# data I/O
#data_train = open('input_train.txt', 'r').read() # should be simple plain text file
#data_test = open('input_test.txt', 'r').read() # should be simple plain text file
data_train=np.zeros((5,100000))
data_test=np.zeros((5,100000))

data=np.loadtxt('input_train.txt', delimiter=',', usecols=range(500000), unpack=True)

data_train[0,:]=data[0:100000]
data_train[1,:]=data[100000:200000]
data_train[2,:]=data[200000:300000]
data_train[3,:]=data[300000:400000]
data_train[4,:]=data[400000:500000]

data=np.loadtxt('input_test.txt', delimiter=',', usecols=range(500000), unpack=True)

data_test[0,:]=data[0:100000]
data_test[1,:]=data[100000:200000]
data_test[2,:]=data[200000:300000]
data_test[3,:]=data[300000:400000]
data_test[4,:]=data[400000:500000]

# hyperParameters
data_size=100000
input_window=50
input_classes=1
output_classes=5
input_size=input_window*input_classes
output_size=output_classes

threshold=100000
hidden_size=256
learning_rate=5e-2
iterations=2*threshold
steps=30
n=0
p=1
epoch=0
loss=0
ave_loss=0
alpha=0.01

print 'Data has %d values' % (data_size)

# model parameters
Wf = np.random.randn(hidden_size+input_size, hidden_size)*0.01 # input to hidden
Wi = np.random.randn(hidden_size+input_size, hidden_size)*0.01 # input to hidden
Wc = np.random.randn(hidden_size+input_size, hidden_size)*0.01 # input to hidden
Wo = np.random.randn(hidden_size+input_size, hidden_size)*0.01 # input to hidden
Wy = np.random.randn(hidden_size, output_size)*0.01 # hidden to output
bf = np.zeros((1,hidden_size)) # hidden bias
bi = np.zeros((1,hidden_size)) # hidden bias
bc = np.zeros((1,hidden_size)) # hidden bias
bo = np.zeros((1,hidden_size)) # hidden bias
by = np.zeros((1,output_size)) # output bias

#forward and backward lstm
def lossFun(inputs, targets,out_index):
  xs, xss, hf, hi, ho, hc, c, h, ys, ps= {}, {}, {}, {}, {}, {}, {}, {},{}, {}
  h[-1] = np.zeros((1,hidden_size))
  c[-1] = np.zeros((1,hidden_size))
  loss = 0
  o=np.random.permutation(data_size-input_window*steps)
 #Forward 
  for t in xrange(steps):
    loss=0
    xs = np.zeros((1, input_window)) # encode the input representation
    xs = inputs_train[o[0]-1+input_window*t:o[0]-1+input_window*t+input_window]
    xs=np.array(xs).reshape(1, input_window)
    xss[t]=np.column_stack((h[t-1], xs))
    hf[t] = 1/(1+np.exp(-(np.dot(xss[t], Wf)+ bf))) # hidden state
    hi[t] = 1/(1+np.exp(-(np.dot(xss[t], Wi)+ bi))) # hidden state
    ho[t] = 1/(1+np.exp(-(np.dot(xss[t], Wo)+ bo))) # hidden state
    hc[t] = np.tanh(np.dot(xss[t], Wc)+ bc) # hidden state
    c[t]=hf[t]*c[t-1]+hi[t]*hc[t]
    h[t]=ho[t]*np.tanh(c[t])
    ys[t] = np.dot(h[t],Wy) + by # unnormalized log probabilities
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities
    loss_temp=ps[t]
    loss += -np.log(loss_temp[:,out_index]) # softmax (cross-entropy loss)
  dWf, dWi, dWc, dWo , dWy, dh, dho, dc, dhf, dhi, dhc, dXf, dXi, dXo, dXc= np.zeros_like(Wf), np.zeros_like(Wi), np.zeros_like(Wc), np.zeros_like(Wo), np.zeros_like(Wy), np.zeros_like(Wf), np.zeros_like(Wf), np.zeros_like(Wf), np.zeros_like(Wf), np.zeros_like(Wf), np.zeros_like(Wf), np.zeros_like(Wf), np.zeros_like(Wf), np.zeros_like(Wf), np.zeros_like(Wf)
  dbf, dbi, dbc, dbo, dby= np.zeros_like(bf), np.zeros_like(bi), np.zeros_like(bc), np.zeros_like(bo), np.zeros_like(by)
  dh_next, dc_next = np.zeros_like(h[0]), np.zeros_like(c[0])

#Backward
  for t in reversed(xrange(steps)):
    dy = np.copy(ps[t])
    dy_temp=dy.T
    dy_temp[out_index] -= 1 # backprop into y
    dy=dy_temp.T
    dWy += np.dot(h[t].T,dy)
    dby += dy
    dh = np.dot(dy, Wy.T) + dh_next # backprop into h
    dho = np.tanh(c[t]) * dh
    dho = ho[t]*(1-ho[t]) * dho
    dc = ho[t] * dh * (1-np.tanh(c[t])*np.tanh(c[t]))
    dc = dc + dc_next
    dhf = hf[t]*(1-hf[t]) * c[t] * dc
    dhi = hc[t] * dc
    dhi = hi[t]*(1-hi[t]) * dhi
    dhc = hi[t] * dc
    dhc = (1-hc[t]*hc[t]) * dhc
    dWf += np.dot(xss[t].T, dhf)
    dbf += dhf
    dXf = np.dot(dhf, Wf.T)
    dWi += np.dot(xss[t].T, dhi)
    dbi += dhi
    dXi = np.dot(dhi, Wi.T)
    dWo += np.dot(xss[t].T, dho)
    dbo += dho
    dXo = np.dot(dho, Wo.T)
    dWc += np.dot(xss[t].T, dhc)
    dbc += dhc
    dXc = np.dot(dhc, Wc.T)
    # As X was used in multiple gates, the gradient must be accumulated here
    dX = dXo + dXc + dXi + dXf
    # Split the concatenated X, so that we get our gradient of h_old
    dh_next = dX[:, :hidden_size]
    # Gradient for c_old in c = hf * c_old + hi * hc
    dc_next = hf[t] * dc

  for dparam in [dWf, dWi, dWc, dWo , dWy, dbf, dbi, dbc, dbo, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWf, dWi, dWc, dWo , dWy, dbf, dbi, dbc, dbo, dby

# n times sampling test of the input
def sample(seed_ix, n):

  correct=0
  for d in xrange (n):
          hh = np.zeros((1, hidden_size)) # reset RNN memory
          cc = np.zeros((1, hidden_size)) # reset RNN memory
          zz=np.random.permutation(output_classes)
          x=seed_ix [zz[1],:]
          kk=np.random.permutation(data_size-input_window*steps)
	  for t in xrange(steps):
	    xx=x[kk[0]-1+input_window*t:kk[0]-1+input_window*t+input_window]
            xx=np.array(xx).reshape(1, input_window)
	    xx = np.column_stack((hh, xx))
	    hff = 1/(1+np.exp(-(np.dot(xx, Wf)+ bf))) # hidden state
	    hii= 1/(1+np.exp(-(np.dot(xx, Wi)+ bi))) # hidden state
	    hoo = 1/(1+np.exp(-(np.dot(xx, Wo)+ bo))) # hidden state
	    hcc= np.tanh(np.dot(xx, Wc)+ bc) # hidden state
	    cc=hff*cc+hii*hcc
	    hh=hoo*np.tanh(cc)
	    yy= np.dot(hh,Wy) + by # unnormalized log probabilities for next chars
	    p= np.exp(yy) / np.sum(np.exp(yy)) # probabilities for next chars
	    ix = np.random.choice(range(output_classes), p=p.ravel())
	  if ix==zz[1]:
	    correct=correct+1
  return correct

n= 0
mWf, mWi, mWc, mWo , mWy= np.zeros_like(Wf), np.zeros_like(Wi), np.zeros_like(Wc), np.zeros_like(Wo), np.zeros_like(Wy)
mbf, mbi, mbc, mbo, mby= np.zeros_like(bf), np.zeros_like(bi), np.zeros_like(bc), np.zeros_like(bo), np.zeros_like(by)
smooth_loss = 1 # loss at iteration 0
for n in xrange(iterations):
  # initial reset
  if n==0: 
    hprev = np.zeros((1, hidden_size)) # reset RNN memory
    cprev = np.zeros((1, hidden_size)) # reset RNN memory
    correct=0

  m=np.random.permutation(output_classes)
  inputs_train = data_train [m[0],:]
  inputs_test = data_test
  targets_train = np.zeros_like(by)
  targets_train[0,m[0]]=1
  # sample from the model now and then
  if n % 1000 == 0:
    out = sample(inputs_test, 1000)
    print 'accuracy %f' % (float(out)/(1000)*100) 

  # forward one class through the net and fetch gradient
  loss, dWf, dWi, dWc, dWo , dWy, dbf, dbi, dbc, dbo, dby= lossFun(inputs_train, targets_train, m[0])
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress
  
  # perform parameter update with rmsprop
  for param, dparam, mem in zip([Wf, Wi, Wc, Wo , Wy, bf, bi, bc, bo, by], 
                                [dWf, dWi, dWc, dWo , dWy, dbf, dbi, dbc, dbo, dby], 
                                [mWf, mWi, mWc, mWo , mWy, mbf, mbi, mbc, mbo, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  n += 1 # iteration counter 

