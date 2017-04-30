/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.ensor.fftmusings.io;

import java.io.Closeable;
import java.util.Iterator;

/**
 *
 * @author jona
 */
public interface ICloseableIterator<T> extends Closeable, Iterator<T> {
    
}
