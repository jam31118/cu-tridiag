#ifndef _HELPER_H_
#define _HELPER_H_

template <typename m_t>
void swap_pointers(m_t **p1, m_t **p2) {
  m_t *temp_p = *p1;
  *p1 = *p2;
  *p2 = temp_p;
}

#endif // _HELPER_H_
