/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.io;

import java.io.IOException;
import java.util.List;

/**
 *
 * @author jona
 */
public class CompositeIterator<T> implements ICloseableIterator<T> {

    private final List<ICloseableIterator<T>> mChildren;
    private ICloseableIterator<T> mCurrentIterator;
    
    public CompositeIterator(List<ICloseableIterator<T>> aChildren) {
        mChildren = aChildren;
        if (mChildren == null || mChildren.isEmpty()) {
            mCurrentIterator = null;
        }
        else {
            mCurrentIterator = mChildren.get(0);
        }
    }
    
    @Override
    public boolean hasNext() {
        if (mCurrentIterator != null) {
            if (mCurrentIterator.hasNext()) return true;
            else {
                try {
                    mCurrentIterator.close();
                } catch (IOException ex) {
                    throw new RuntimeException("IO Exception closing", ex);
                }
                mChildren.remove(0);
                if (mChildren.isEmpty()) {
                    return false;
                }
                mCurrentIterator = mChildren.get(0);
                return mCurrentIterator.hasNext();
            }
        }
        return false;
    }

    @Override
    public T next() {
        return mCurrentIterator.next();
    }

    @Override
    public void close() throws IOException {
        mCurrentIterator.close();
    }
    
}
